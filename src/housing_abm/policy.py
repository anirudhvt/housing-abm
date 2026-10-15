"""Macroprudential policies (Sections 3.7, 6.2): hard LTV limits that override
mortgage_terms.yaml, and soft LTI limits that let a share of bids exceed the cap."""
import yaml


def load_policies(model, policy_paths):
    """Load policy YAMLs onto the model and apply any LTV overrides."""
    model.policies = []
    for path in policy_paths or []:
        with open(path) as f:
            model.policies.extend(yaml.safe_load(f))
    # invalidate the by-type index built in investor_restrictions
    model._policy_type_index = None
    _apply_hard_ltv_overrides(model)


def _apply_hard_ltv_overrides(model):
    for policy in model.policies:
        if policy["type"] == "ltv_limit":
            loan_type = policy["loan_type"]
            model.mortgage_terms[loan_type]["max_ltv"] = policy["hard_limit"]


def enforce_lti_policies(model):
    """Before matching, cap over-limit bids at the hard LTI, exempting a random
    soft_allowance share of affected bids."""
    lti_policies = [p for p in model.policies if p["type"] == "lti_limit"]
    if not lti_policies:
        return

    for policy in lti_policies:
        loan_type = policy["loan_type"]
        hard_limit = policy["hard_limit"]
        allowance = policy.get("soft_allowance", 0.0)

        affected = [
            b for b in model._ownership_bid_queue if b["agent"].LOAN_TYPE == loan_type
        ]
        if not affected:
            continue

        def implied_lti(bid):
            loan = bid["max_price"] - bid["down_payment"]
            annual_income = bid["agent"].income * 12.0
            return loan / annual_income if annual_income > 0 else 0.0

        over_limit = [b for b in affected if implied_lti(b) > hard_limit]
        if not over_limit:
            continue

        model.random_gen.shuffle(over_limit)
        n_exempt = int(round(len(affected) * allowance))
        capped = over_limit[n_exempt:]  # the first n_exempt keep their bid

        for bid in capped:
            max_loan = hard_limit * bid["agent"].income * 12.0
            bid["max_price"] = bid["down_payment"] + max_loan
