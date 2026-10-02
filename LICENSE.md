# OPEN ENCODING ATTRIBUTION LICENSE — Version 1.0

>
> **This is a SOURCE-AVAILABLE license, not an open-source license.** Because the
> attribution duty in §5 applies only to commercial (for-profit) use, the License
> discriminates by field of endeavor and does not meet clause 6 of the Open Source
> Definition. Do not describe software under this License as "open source."

---

## Gist: what you can do, by use case

*Informational only — the numbered Terms below control. "Credit (§5)" = display the short
Attribution Information from Exhibit A in one allowed location. "Notices (§4)" = keep
existing notices and carry forward the `NOTICE` file. Note that §4 (notices) applies to
**any** distribution, while §5 (visible credit) applies only to **commercial** use.*

| Your use case                                                  | Allowed? | Royalty? | Modify / fork? | Show credit (§5)   | Keep notices (§4)   | Patent grant (§3) |
|----------------------------------------------------------------|----------|----------|----------------|--------------------|---------------------|-------------------|
| **Personal / hobby / private** use                             | ✅ Yes    | None     | ✅ Yes          | ❌ No (safe harbor) | ✅ if you distribute | ✅ Yes             |
| **Nonprofit, academic, research, or government** use           | ✅ Yes    | None     | ✅ Yes          | ❌ No (safe harbor) | ✅ if you distribute | ✅ Yes             |
| **Internal for-profit** use, nothing exposed to third parties  | ✅ Yes    | None     | ✅ Yes          | ❌ No ¹             | ✅ if you distribute | ✅ Yes             |
| **Ship in a commercial / for-profit product**                  | ✅ Yes    | None     | ✅ Yes          | ✅ **Yes**          | ✅ Yes               | ✅ Yes             |
| **Run a commercial / for-profit service** (the "YouTube" case) | ✅ Yes    | None     | ✅ Yes          | ✅ **Yes** ²        | ✅ if you distribute | ✅ Yes             |
| **Fork into a copyleft / GPL project**                         | ⚠️ See ³ | None     | ✅ Yes          | per row above      | ✅ Yes               | ✅ Yes             |
| **Sue a contributor over their patents**                       | —        | —        | —              | —                  | —                   | ❌ **Revoked** ⁴   |

**¹ Internal-only carve-out.** A purely internal commercial tool shows the credit to no
audience, so §5 does not fire. To require the credit for *all* for-profit use (including
internal), see the one-line edit noted inside §5.

**² Commercial network-use trigger.** Operating the Software as a commercial service
("Commercial External Deployment") fires the credit requirement *even though you distribute
nothing*. This is the clause that catches a hosted, for-profit service built on the Software.

**³ Copyleft compatibility.** §6 permits forking into a more restrictive license, but the §5
credit is an added restriction the GNU GPL will not accept → **not GPL-compatible**. (This is
on top of the source-available status above.)

**⁴ Defensive termination.** Filing patent litigation over the Software terminates your patent
license under §3.

> ⚠️ **Codec patents are separate.** This license grants only patents your *contributors*
> hold. It does **not** grant rights to third-party codec pools (AAC, H.264/AVC, H.265/HEVC,
> VVC, Via LA, Access Advance). If your use encodes/decodes a patented format, you may owe
> those pools independently. No software license can change that — get counsel.

> ⚠️ **No warranty, no liability** in every case (§8, §9).

---

## Plain-language summary (not part of the license; the numbered terms control)

- Do almost anything: use, copy, modify, fork, keep your fork private, distribute, and use it
  commercially. No royalties, ever. (§2)
- Each contributor grants you a patent license for their own contributions, lost only if you
  sue them over those patents. (§3)
- If you **redistribute** it, keep the existing notices and carry forward the `NOTICE` file. (§4)
- If you use it **commercially in something third parties can see** — a product you ship or a
  for-profit service you run — you must show a short, one-line credit for the encoding
  technology somewhere a person can find it (About/Credits/Licenses screen, docs, or splash).
  Personal, hobby, nonprofit, academic, and government use are exempt. (§5)
- You may relicense your modifications, including under copyleft, as long as the §4 notices and
  the §5 credit obligation survive. (§6)
- No warranty. No liability. (§8, §9)

---

## TERMS AND CONDITIONS

### 1. Definitions

**"License"** means the terms and conditions of this document.

**"Software"** means the work of authorship made available under this License, in source or
object form, including any portion thereof.

**"Licensor"** means the copyright owner or an entity authorized by the copyright owner that
is granting the License.

**"You"** (or **"Your"**) means any individual or legal entity exercising permissions granted
by this License.

**"Contributor"** means the Licensor and any individual or legal entity on behalf of whom a
Contribution is received and incorporated into the Software.

**"Contribution"** means any work of authorship intentionally submitted for inclusion in the
Software by its copyright owner.

**"Derivative Work"** means any work that is based on (or derived from) the Software and for
which the modifications represent, as a whole, an original work of authorship.

**"Distribute"** means to make the Software or a Derivative Work available to any third party,
in source or object form, by any means.

**"External Deployment"** means to use, operate, or make the Software or a Derivative Work
available to any third party as part of a service over a network (including a website, API, or
hosted application), whether or not the Software itself is Distributed.

**"Commercial Use"** means use of the Software or a Derivative Work primarily intended for or
directed toward commercial advantage or monetary compensation, including incorporating it into
a product or service that You sell, license for a fee, monetize, or operate for profit.

**"Personal Use"** means use by an individual for research, experiment, study, private
entertainment, hobby projects, or amateur pursuits, in each case without any anticipated
commercial application. Personal Use is never Commercial Use.

**"Noncommercial Organization"** means a charitable organization, educational institution,
public research organization, public safety or health organization, environmental protection
organization, or government institution. Use by a Noncommercial Organization, for its own
purposes, is not Commercial Use regardless of the source of its funding.

**"Attribution Information"** means the short credit string, optional URL, and optional
graphic specified in EXHIBIT A.

**"NOTICE"** means the text file (if any) named `NOTICE` distributed with the Software.

### 2. Copyright grant

Subject to the terms of this License, each Contributor grants You a worldwide, royalty-free,
non-exclusive, perpetual, and irrevocable (except as stated in §3 and §7) license to use,
reproduce, modify, prepare Derivative Works of, publicly display, publicly perform,
sublicense, and Distribute the Software and such Derivative Works, for any purpose, including
commercial and for-profit purposes, and including private use of modified or unmodified forks
that are never Distributed.

### 3. Patent grant and defensive termination

Subject to the terms of this License, each Contributor grants You a worldwide, royalty-free,
non-exclusive, perpetual, and (except as stated below) irrevocable patent license to make,
have made, use, offer to sell, sell, import, and otherwise transfer the Software, where such
license applies only to those patent claims licensable by that Contributor that are
necessarily infringed by their Contribution alone or by the combination of their Contribution
with the Software to which it was submitted.

This is a patent license (an affirmative grant plus a covenant not to assert the licensed
claims against You), which is broader and more reliable than a bare "waiver."

If You institute patent litigation against any entity (including a cross-claim or
counterclaim) alleging that the Software or a Contribution incorporated within it constitutes
direct or contributory patent infringement, then any patent licenses granted to You under this
License for that Software terminate as of the date such litigation is filed.

### 4. Conditions on redistribution

You may Distribute the Software or any Derivative Work in source or object form (whether or not
for Commercial Use), provided that You meet ALL of the following:

(a) You give any other recipient a copy of this License;

(b) You cause any modified files to carry prominent notices stating that You changed them;

(c) You retain, in the source form of any Derivative Work, all copyright, patent, trademark,
and attribution notices from the source form of the Software, excluding only notices that do
not pertain to any part of the Derivative Work; and

(d) If the Software includes a `NOTICE` file, then any Derivative Work You Distribute must
include a readable copy of the attribution notices contained in that `NOTICE` file (excluding
notices that do not pertain to any part of the Derivative Work), in at least one of: a `NOTICE`
text file distributed with the Derivative Work; the source form or documentation; or a display
generated by the Derivative Work wherever such third-party notices normally appear.

### 5. Commercial attribution requirement (the "awareness" clause)

The purpose of this Section is to ensure that, when the Software is used commercially, the
encoding technology it embodies remains visibly credited, so that end users and the public are
aware of the technology on which a commercial product or service depends.

**Scope.** This Section applies ONLY to Commercial Use. If Your use is not Commercial Use —
including any Personal Use, or any use by a Noncommercial Organization as defined in §1 — this
Section imposes no obligation on You. (Your obligations under §4 are unaffected.)

**Requirement.** If You make Commercial Use of the Software or a Derivative Work, AND that use
reaches third parties (whether by Distributing it or by External Deployment), then You MUST
display the Attribution Information specified in EXHIBIT A in at least ONE of the following
locations, chosen by You:

(a) within an "About", "Credits", "Acknowledgements", "Licenses", or equivalent screen, page,
or dialog reachable through the normal user interface of Your product or service; or

(b) within the product's user-facing documentation (including an online documentation site or
a README distributed with the product); or

(c) on a splash, startup, or loading screen shown to end users; or

(d) if Your product genuinely exposes no user interface, documentation, or comparable surface
(for example, an embedded or headless library), within a plain-text notice file distributed or
made available with the product.

**Constraints, to keep this requirement light:**

- The required credit text MUST NOT exceed twelve (12) words plus an optional URL (see
  EXHIBIT A). The Licensor may not demand more.
- You need not display it more prominently than other third-party credits of comparable nature
  in the same surface.
- You may combine it with other attribution notices.
- Where multiple works licensed under this License are combined, a single consolidated credit
  listing each by name satisfies this Section; the requirements do not stack.

**Optional broadening.** To require the credit for *all* Commercial Use, including purely
internal commercial use with no third-party exposure, delete the words "AND that use reaches
third parties (whether by Distributing it or by External Deployment)" above.

**Status note.** Because this Section conditions an obligation on commercial use, the License
is source-available rather than open source (Open Source Definition, clause 6), and is not
GPL-compatible. To make it open-source-eligible, this Section must apply equally to all users
regardless of commercial status.

### 6. Downstream licensing and copyleft

You may license Your Contributions and Derivative Works to recipients under this License, or
under any other license of Your choosing (including a copyleft or otherwise more restrictive
license), PROVIDED that:

(a) the conditions of §4 and §5 continue to be satisfied with respect to the portions of the
Software that originated under this License; and

(b) You do not purport to grant rights in the original Software beyond those You actually
received under this License.

Forking into a copyleft project is permitted, but the §4 notice obligations and the §5
commercial-attribution obligation travel with the original code and cannot be stripped by
relicensing.

### 7. Trademarks

This License does not grant permission to use the trade names, trademarks, service marks,
product names, or logos of any Contributor, except as required for reasonable and customary
use in describing the origin of the Software and reproducing the `NOTICE` file. The graphic in
EXHIBIT A, if any, is licensed solely for the purpose of satisfying §5 and for no other
purpose.

### 8. Disclaimer of warranty

THE SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OR CONDITIONS OF
TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY
RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING THE SOFTWARE AND
ASSUME ANY RISKS ASSOCIATED THEREWITH.

### 9. Limitation of liability

IN NO EVENT AND UNDER NO LEGAL THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR
OTHERWISE, SHALL ANY CONTRIBUTOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT, INDIRECT,
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES OF ANY CHARACTER ARISING AS A RESULT OF THIS
LICENSE OR THE USE OF THE SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

### 10. Acceptance

You accept and agree to this License by exercising any permission granted under §2 or §3. If
You do not accept this License, You have no rights to the Software.

---

## EXHIBIT A — Attribution Information (fill in; this is what §5 requires)

- **Attribution credit text** (REQUIRED; max 12 words): e.g. `Video encoding powered by Rivet`
- **Attribution URL** (OPTIONAL): e.g. `https://github.com/rivet-transcoder/rivet`
- **Attribution graphic** (OPTIONAL; if provided, governed by §7): `[ none / path-to-logo ]`

---

## How to apply this license

1. Rename the license (title + identifier) so it is not confused with Apache, CPAL, PolyForm,
   or any other named license. Do not call it "open source."
2. Put this file in your repository root as `LICENSE` (or `LICENSE.md`).
3. Add a `NOTICE` file containing the credit you want propagated (§4(d)).
4. Fill in EXHIBIT A.
5. Add a short header to source files, for example:

   ```
   Copyright [yyyy] [name of copyright owner]
   Licensed under the Open Encoding Attribution License, Version 1.0.
   You may not use this file except in compliance with the License.
   See the LICENSE file for terms, including the commercial attribution
   requirement in Section 5.
   ```

6. **Have a qualified IP attorney review before relying on it** — especially the §1 definition
   of "Commercial Use," §3 (patent), §5 (commercial trigger), and the codec-patent advisory.

---

## Suggested `NOTICE` file contents

```
[Project Name]
Copyright [yyyy] [Author]

This product includes [Project Name] encoding technology (https://...).
Commercial products and services that use it must display the
attribution in Section 5 of its license.
```