#!/usr/bin/env bash
# apply_protection.sh
# ===================================================================
# STATUS: NON-AUTHORITATIVE REFERENCE. DO NOT RUN. DISABLED 2026-08-22.
# ===================================================================
#
# This script no longer applies anything. It exits non-zero before it
# touches the GitHub API. It is kept only as a record of intent and as
# a transcription of the live ruleset for disaster recovery.
#
# SOURCE OF TRUTH
#   The live branch protection for Gavin-Borges/SESTRAV is ruleset
#   id 16846770, name "Protect Main Branch", configured and maintained
#   through the GitHub web UI (created 2026-05-25, last updated
#   2026-08-22T19:14:42-04:00 - re-read from the live API 2026-08-24;
#   this line previously said 2026-07-26 and was a month stale, which
#   matters because the freshness of this whole transcription is
#   asserted by that date). This file is a downstream transcription of it, not
#   its definition. Change protection in the UI, then update the
#   REFERENCE PAYLOAD block below to match.
#
#   Read the live state at any time (read-only):
#     gh api repos/Gavin-Borges/SESTRAV/rulesets/16846770
#
# WHY IT WAS DISABLED (lockout hazard, verified live 2026-08-22)
#   Repository rulesets are ADDITIVE: GitHub evaluates every active
#   ruleset and the most restrictive rule wins. So the danger here was
#   never that running this would weaken or disable protection. It is
#   that running it would ADD a second, wrong ruleset whose required
#   status check never reports, permanently blocking every merge.
#
#   1. NAME MISMATCH -> ACCIDENTAL CREATE.
#      The old lookup selected `.name == "Protect main"`. The live
#      ruleset is named "Protect Main Branch", so the lookup missed,
#      METHOD fell through to POST, and the script CREATED A SECOND
#      RULESET instead of updating the existing one.
#
#   2. PHANTOM REQUIRED CHECK -> PERMANENT PENDING.
#      The old payload required the context
#      "SESTRAV CI / test (3.13)". No such check run exists. The
#      reporting context is "test (3.13)" (verified against the check
#      runs on PR #283 head 21fed6a and on main HEAD 32abe8c; every
#      SESTRAV check reports its bare job name, never an
#      "SESTRAV CI / " prefix). A required check that never reports
#      stays pending forever, so the added ruleset would have blocked
#      the merge button on every PR indefinitely.
#
#   3. MISSING RULES AND CONTEXTS.
#      The old payload had no `code_scanning` rule and named only one
#      of the five live required contexts.
#
#   4. WRONG BYPASS ACTOR.
#      The old payload added the owner as actor_type "User" with the
#      numeric user id. Live uses actor_type "RepositoryRole" with
#      actor_id 5 (repository admin).
#
#   5. EXTRA RULE NOT IN LIVE.
#      The old payload added `require_linear_history`, which the live
#      ruleset does not have. Because rulesets are additive, adding it
#      would have started rejecting the merge commits this repo
#      actually uses (see merge commits a1a8fe8, d00754e).
#
# WHY THIS WAS NOT SIMPLY CORRECTED AND LEFT RUNNABLE
#   Updating a ruleset is a full replace (PUT), so any field the
#   payload omits is reset. The live ruleset carries
#   `require_extra_approval_for_unattributed_changes: true` inside its
#   pull_request parameters. That field is NOT in the documented
#   request schema for PUT /repos/{owner}/{repo}/rulesets/{id}; GitHub
#   documents the corresponding UI setting ("Require an additional
#   approval for unattributed Copilot pull requests") but does not
#   state that it is API-writable.
#
#   That leaves no verifiably safe choice. Sending the field may be
#   rejected or silently dropped; omitting it may silently reset a
#   protection that is currently on. Distinguishing those outcomes
#   requires performing the very write this file is meant to make
#   safe, so it cannot be settled read-only.
#
#   More broadly: the presence of one live field outside the
#   documented write schema means a GET response can carry fields the
#   write schema does not cover. There is therefore no read-only way
#   to prove that a hand-built full-replace payload drops nothing.
#   The uncertainty is open-ended, not a single known gap.
#
# DISASTER RECOVERY
#   If ruleset 16846770 is ever lost, recreate it through the GitHub
#   UI (Settings > Rules > Rulesets) using the REFERENCE PAYLOAD below
#   as the checklist, then diff the result against this block:
#     gh api repos/Gavin-Borges/SESTRAV/rulesets/<new_id>
#   Do not paste the payload into a write call without first
#   confirming the current request schema for every field, especially
#   `require_extra_approval_for_unattributed_changes`.
#
# ===================================================================
# REFERENCE PAYLOAD - live state as of 2026-08-22, read-only verified.
# Inert documentation. Read-only fields (id, node_id, _links,
# created_at, updated_at, source, source_type, current_user_can_bypass)
# are omitted because they are not writable. Strip the leading "# "
# from each line to recover valid JSON.
#
# {
#   "name": "Protect Main Branch",
#   "target": "branch",
#   "enforcement": "active",
#   "conditions": {
#     "ref_name": {
#       "include": ["refs/heads/main"],
#       "exclude": []
#     }
#   },
#   "bypass_actors": [
#     {
#       "actor_id": 5,
#       "actor_type": "RepositoryRole",
#       "bypass_mode": "always"
#     }
#   ],
#   "rules": [
#     { "type": "deletion" },
#     { "type": "non_fast_forward" },
#     {
#       "type": "pull_request",
#       "parameters": {
#         "required_approving_review_count": 0,
#         "dismiss_stale_reviews_on_push": true,
#         "require_code_owner_review": true,
#         "require_last_push_approval": false,
#         "required_review_thread_resolution": false,
#         "required_reviewers": [],
#         "allowed_merge_methods": ["merge", "squash", "rebase"],
#         "require_extra_approval_for_unattributed_changes": true
#       }
#     },
#     {
#       "type": "code_scanning",
#       "parameters": {
#         "code_scanning_tools": [
#           {
#             "tool": "CodeQL",
#             "security_alerts_threshold": "all",
#             "alerts_threshold": "errors_and_warnings"
#           }
#         ]
#       }
#     },
#     {
#       "type": "required_status_checks",
#       "parameters": {
#         "strict_required_status_checks_policy": true,
#         "do_not_enforce_on_create": false,
#         "required_status_checks": [
#           { "context": "test (3.13)",          "integration_id": 15368 },
#           { "context": "Require human review", "integration_id": 15368 },
#           { "context": "check_dco",            "integration_id": 15368 },
#           { "context": "Cited commits resolve","integration_id": 15368 },
#           { "context": "Cited lines still hold their content",
#                                                "integration_id": 15368 }
#         ]
#       }
#     }
#   ]
# }
#
# integration_id 15368 is the GitHub Actions app; verified from the
# check runs on PR #283 head 21fed6a, where ALL FOUR required contexts
# THEN IN FORCE reported with app.id 15368 / app.slug "github-actions".
# The fifth context, "Cited lines still hold their content", was added
# to the ruleset on 2026-08-22 after that observation; its
# integration_id was read directly from the live ruleset on 2026-08-24
# and is also 15368.
#
# Cite a pull-request head here, not a main-branch commit. Measured
# against main HEAD's check runs on 2026-08-24, exactly TWO of the five
# required contexts report on a push to main: "test (3.13)" and
# "Cited lines still hold their content" - the latter because
# doc_line_citations.yml carries BOTH a pull_request and a push
# trigger. "Require human review", "check_dco" and "Cited commits
# resolve" are pull_request-only and are absent from main HEAD's check
# runs entirely. A main-branch commit therefore evidences two of the
# five entries below, not five.
#
# An earlier version of this note said only "test (3.13)" reports on a
# push to main. That was false, and it was false in the direction that
# made this file look more carefully reasoned than it was.
#
# The five required contexts are documented in SECURITY.md under
# "Vulnerability Triage & Remediation Policy" / "CI gate map". Keep
# that table, this block, and the live ruleset in agreement.
#
# That instruction was not followed when the fifth context was added on
# 2026-08-22: this block, SECURITY.md, docs/SCORECARD_REMEDIATION.md,
# docs/security_compliance.md, .github/workflows/security.yml and one
# CHANGELOG line all still said four. Corrected together 2026-08-24.
# Six sites is what "keep in agreement" actually costs - if a seventh
# is ever added, add it here too rather than to one document.
# ===================================================================

set -uo pipefail

cat >&2 <<'GUARD'
apply_protection.sh is DISABLED and applies nothing.

Branch protection for Gavin-Borges/SESTRAV is ruleset 16846770
("Protect Main Branch"), managed in the GitHub web UI. That is the
source of truth.

This script's old payload looked up the wrong ruleset name, so it
created a second ruleset rather than updating the existing one, and
that ruleset required a status check ("SESTRAV CI / test (3.13)")
that nothing ever reports. Rulesets are additive, so the result was a
merge button blocked on a forever-pending check, on every PR.

Read the live ruleset instead (read-only):
  gh api repos/Gavin-Borges/SESTRAV/rulesets/16846770

To change protection, use Settings > Rules > Rulesets in the GitHub
UI, then update the REFERENCE PAYLOAD block in this file to match.
GUARD

exit 1
