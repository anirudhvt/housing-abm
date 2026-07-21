"""Section 3.7 + 6.2: Hard LTV limits (overriding mortgage_terms.yaml)
 and 'soft' LTI limits (a hard cap with some allowance of share of new mortgages permitted above it

ICR policies and state-contingent triggers (Section 3.7 item 3, e.g. "LTV 85% if
credit growth exceeds X") aren't implemented yet."""

import yaml


def load_policies(model, policy_paths): #grab policies from config
    model.policies = []
    for path in (policy_paths or []):
        with open(path) as f:
            model.policies.extend(yaml.safe_load(f))
    _apply_hard_ltv_overrides(model)


def _apply_hard_ltv_overrides(model):
    for policy in model.policies:
        if policy["type"] == "ltv_limit":
            loan_type = policy["loan_type"]
            model.mortgage_terms[loan_type]["max_ltv"] = policy["hard_limit"] #edit params


def enforce_lti_policies(model):
    """
    run before ownership market
    clamp worst offenders down to hard limit
    exempt some random bids - clear at original LTI (soft_allowance)"""
    lti_policies = [p for p in model.policies if p["type"] == "lti_limit"]
    if not lti_policies: #no lti policies
        return

    for policy in lti_policies:
        #grab details about the policy
        loan_type = policy["loan_type"]
        hard_limit = policy["hard_limit"]
        allowance = policy.get("soft_allowance", 0.0)


        #only affects certain loans
        affected = [b for b in model._ownership_bid_queue if b["agent"].LOAN_TYPE == loan_type]
        if not affected:
            continue

        def implied_lti(bid):
            """Grab LTI of agent"""
            loan = bid["max_price"] - bid["down_payment"]
            annual_income = bid["agent"].income * 12.0
            return loan / annual_income if annual_income > 0 else 0.0


        over_limit = [b for b in affected if implied_lti(b) > hard_limit]
        if not over_limit: #not affected
            continue
        
        #affected by policy
        model.random_gen.shuffle(over_limit)
        n_exempt = int(round(len(affected) * allowance))
        capped = over_limit[n_exempt:]  #first n_exempt keep their original (higher) bid

        for bid in capped:
            max_loan = hard_limit * bid["agent"].income * 12.0
            bid["max_price"] = bid["down_payment"] + max_loan