"""Agentic-outreach Temporal activities.

The LLM-chain activities for the outreach conversation:
    - mine_reviews_activity              — Agent A (review-hook miner)
    - fill_opener_template_activity      — Agent B (opener writer)
    - handle_reply_qualifier_activity    — Agent C (qualifier / free-site pitch)
    - interpret_response_activity        — Agent E (router / STOP gate)

Business-wiring / persistence activities (outbound send, site reveal, CRM
persistence, contact enrichment) are referenced by the workflow BY STRING NAME
only and are intentionally not shipped in the showcase.
"""
