//! A small pool of worker threads that run one borrowed closure at a time,
//! so that a frame's work can be spread over threads without starting
//! threads for every frame (on some systems that costs as much as decoding
//! a small frame).

use std::any::Any;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;

/// The closure being run, its lifetime erased (see [`Pool::run`]).
type Task = *const (dyn Fn() + Sync);

struct State {
    /// The closure of the current run, while one is in progress.
    task: Option<SendTask>,
    /// Counts runs, so a worker runs each task once.
    generation: u64,
    /// Workers still inside the current run's closure.
    running: usize,
    /// The first panic of the current run, rethrown by `run`.
    panic: Option<Box<dyn Any + Send>>,
    shutdown: bool,
}

#[derive(Clone, Copy)]
struct SendTask(Task);

// SAFETY: the pointee is `Sync` (so callable from any thread), and `run`
// keeps it alive until every worker has finished with it.
unsafe impl Send for SendTask {}

struct Shared {
    state: Mutex<State>,
    /// Signals workers: a new run, or shutdown.
    start: Condvar,
    /// Signals the caller: a worker finished its part of the run.
    done: Condvar,
}

/// Worker threads (the caller's thread works too).
pub(crate) struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

fn lock(m: &Mutex<State>) -> MutexGuard<'_, State> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Pool {
    /// A pool with `threads - 1` workers: `run` uses `threads` threads.
    pub(crate) fn new(threads: usize) -> Pool {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                task: None,
                generation: 0,
                running: 0,
                panic: None,
                shutdown: false,
            }),
            start: Condvar::new(),
            done: Condvar::new(),
        });
        let workers = (1..threads.max(1))
            .map(|i| {
                let shared = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("vp8-worker-{i}"))
                    .spawn(move || worker(&shared))
                    .expect("spawning a worker thread")
            })
            .collect();
        Pool { shared, workers }
    }

    /// Threads `run` uses, the caller's included.
    pub(crate) fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Calls `f` on every worker and on this thread at once, and returns
    /// when all have returned. A panic in any of them is rethrown here.
    pub(crate) fn run(&self, f: &(dyn Fn() + Sync)) {
        if self.workers.is_empty() {
            f();
            return;
        }
        // SAFETY: the pointer outlives its use: this function does not
        // return (or unwind) before every worker that took the task has
        // finished calling it (`running` back to 0) and the task is
        // cleared, so no worker can reach `f` afterwards.
        let task: Task = unsafe { std::mem::transmute::<&(dyn Fn() + Sync), Task>(f) };
        {
            let mut st = lock(&self.shared.state);
            st.task = Some(SendTask(task));
            st.generation += 1;
            st.running = self.workers.len();
            st.panic = None;
        }
        self.shared.start.notify_all();
        let mine = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        let mut st = lock(&self.shared.state);
        while st.running > 0 {
            st = self.shared.done.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        st.task = None;
        let theirs = st.panic.take();
        drop(st);
        if let Err(p) = mine {
            std::panic::resume_unwind(p);
        }
        if let Some(p) = theirs {
            std::panic::resume_unwind(p);
        }
    }
}

fn worker(shared: &Shared) {
    let mut seen = 0u64;
    loop {
        let task = {
            let mut st = lock(&shared.state);
            loop {
                if st.shutdown {
                    return;
                }
                if st.generation != seen
                    && let Some(t) = st.task
                {
                    seen = st.generation;
                    break t;
                }
                st = shared.start.wait(st).unwrap_or_else(|e| e.into_inner());
            }
        };
        // SAFETY: `run` keeps the closure alive until this worker reports
        // back below.
        let f = unsafe { &*task.0 };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        let mut st = lock(&shared.state);
        if let Err(p) = result {
            st.panic.get_or_insert(p);
        }
        st.running -= 1;
        if st.running == 0 {
            shared.done.notify_all();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        lock(&self.shared.state).shutdown = true;
        self.shared.start.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// A progress counter on a cache line of its own, so that a thread polling
/// one row's progress does not slow down the thread advancing the next.
#[repr(align(128))]
pub(crate) struct Padded(pub AtomicUsize);

impl std::ops::Deref for Padded {
    type Target = AtomicUsize;
    fn deref(&self) -> &AtomicUsize {
        &self.0
    }
}

/// Sets a flag if the thread unwinds, so that threads waiting on its
/// progress give up instead of waiting forever.
pub(crate) struct PanicGuard<'a>(pub &'a AtomicBool);

impl Drop for PanicGuard<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.store(true, Ordering::Relaxed);
        }
    }
}

/// Waits until `counter` reaches `target` (acquiring what was published
/// with it), spinning briefly, then yielding; panics if `poisoned` is set.
pub(crate) fn wait(counter: &AtomicUsize, target: usize, poisoned: &AtomicBool) {
    let mut spins = 0u32;
    while counter.load(Ordering::Acquire) < target {
        if poisoned.load(Ordering::Relaxed) {
            panic!("another thread working on this frame panicked");
        }
        if spins < 200 {
            std::hint::spin_loop();
            spins += 1;
        } else {
            std::thread::yield_now();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_thread_runs_each_task_once() {
        let pool = Pool::new(4);
        for round in 1..50 {
            let n = AtomicUsize::new(0);
            pool.run(&|| {
                n.fetch_add(round, Ordering::Relaxed);
            });
            assert_eq!(n.load(Ordering::Relaxed), 4 * round);
        }
    }

    #[test]
    fn a_panic_reaches_the_caller_and_the_pool_survives() {
        let pool = Pool::new(3);
        let first = AtomicUsize::new(0);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.run(&|| {
                if first.fetch_add(1, Ordering::Relaxed) == 1 {
                    panic!("boom");
                }
            })
        }));
        assert!(r.is_err());
        let n = AtomicUsize::new(0);
        pool.run(&|| {
            n.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(n.load(Ordering::Relaxed), 3);
    }
}
