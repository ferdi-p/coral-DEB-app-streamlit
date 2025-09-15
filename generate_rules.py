# generate_rules.py
from __future__ import annotations
from pathlib import Path
import re
from typing import Dict, Iterable

# ---------------
# Configure these:
# ---------------
# Import your canonical rules dict and list of base symbols.
# Adjust the import path to wherever your rules live now.
from parameters import RULES  # e.g., {'jCP': 'F[jCPm][jL*yCL, ...]/(1+cROS1)', ...}

# Names that are allowed to remain in final expressions (won’t be substituted)
STATE_VARS = {'S', 'H', 'jCP', 'jSG'}                  # your state vector
ENV_VARS   = {'L', 'T', 'Nu', 'X'}                     # environment drivers
PARAMS     = {                                          # bare parameter symbols that should remain
    'yC','nNH','nNS','kCO2','yCL','b','OneOverkROS',
    'Q10','T0','sigmax','steepness','ED50',
    'j0ST0','j0HT0','jXm0','jHGm0','jNm0','jSGm0','jCPm0',
    # add others you actually use
}
ALLOWED = STATE_VARS | ENV_VARS | PARAMS | { 'lambda' }  # 'lambda' if you use it as a param name

# Optional: helper replacements for your special functions F[...] and f[...]
def normalize(expr: str) -> str:
    """Turn Mathematica-ish syntax into Python-callable syntax."""
    s = expr
    # [] -> () for function calls like F[jSGm][a,b] => F(jSGm,a,b)
    s = re.sub(r'([A-Za-z_]\w*)\[(.*?)\]\[(.*?)\]', r'\1(\2,\3)', s)  # F[x][a,b] -> F(x,a,b)
    s = re.sub(r'([A-Za-z_]\w*)\[(.*?)\]', r'\1(\2)', s)              # f[x] -> f(x)
    # powers ^ -> ** (if you still have any)
    s = s.replace('^', '**')
    # greek/Unicode that you mapped to ASCII already (ideally done earlier)
    return s

# A tiny tokenizer for symbol names
SYMBOL = re.compile(r'\b[A-Za-z_]\w*\b')

def find_refs(expr: str) -> Iterable[str]:
    for tok in SYMBOL.findall(expr):
        if tok not in {'for','if','else','max','min','abs'}:
            yield tok

def expand_all(rules: Dict[str,str], allowed: set) -> Dict[str,str]:
    """Iteratively substitute rule names into each other until only allowed symbols remain."""
    # normalize all RHS once
    work = {k: normalize(v) for k, v in rules.items()}

    stable = False
    while not stable:
        stable = True
        for target, rhs in list(work.items()):
            # for every symbol in this RHS that is also a rule key, and not allowed, substitute
            refs = list(find_refs(rhs))
            for r in refs:
                if r in work and r not in allowed and r != target:
                    # replace whole-word r with (work[r])
                    rhs_new = re.sub(rf'\b{re.escape(r)}\b', f'({work[r]})', rhs)
                    if rhs_new != rhs:
                        rhs = rhs_new
                        stable = False
            work[target] = rhs
    # final pass: ensure only allowed symbols remain (or literals/functions)
    for target, rhs in work.items():
        bad = [r for r in set(find_refs(rhs)) if r not in allowed and r not in work]
        if bad:
            print(f"[WARN] {target} still references {bad}. Consider adding to ALLOWED or providing a helper.")
    return work

# --- write out a pure-Python module with functions ---
TEMPLATE_HEADER = '''"""
Auto-generated fluxes and RHS — do not edit by hand.
Regenerate via `python generate_rules.py`.
"""
from __future__ import annotations
import math
# Optionally include pure-python helpers for F and f (your custom operators)

def f(max_, A):
    return (A * max_) / (A + max_) if (A + max_) != 0 else 0.0

def F(max_, A, B):
    # (A B (A+B) max) / (B^2 max + A^2 (B+max) + A B (B+max))
    denom = (B*B*max_) + (A*A*(B+max_)) + (A*B*(B+max_))
    return (A*B*(A+B)*max_) / denom if denom != 0 else 0.0
'''

TEMPLATE_BODY = '''
# Resolved flux expressions (strings), using only allowed symbols
{resolved_block}

def compute_fluxes(S, H, jCP, jSG, L, T, Nu, X, p):
    """
    p: parameter dict-like (p['yC'], p['Q10'], ...)
    Returns a dict of resolved fluxes by name (only those you care about).
    """
    # bind parameters to local names for speed
    {param_binds}

    # state/env locals
    S=float(S); H=float(H); jCP=float(jCP); jSG=float(jSG)
    L=float(L); T=float(T); Nu=float(Nu); X=float(X)

    # derived temperature scalars (example; adjust to your final model)
    e = math.exp(-steepness*(T-ED50))
    sigmoid = sigmax * e / (1.0 + e)
    e0 = math.exp(-steepness*(T0-ED50))
    sigmoid_T0 = sigmax * e0 / (1.0 + e0)
    Beta = sigmoid / sigmoid_T0
    Alpha = pow(Q10, (T - T0) / 10.0)

    # rescale temp-dependent max rates (example; align with your rules)
    j0ST = Alpha * j0ST0
    j0HT = Alpha * j0HT0
    jXm  = Alpha * jXm0
    jHGm = Alpha * jHGm0
    jNm  = Alpha * jNm0
    jSGm = Alpha * jSGm0
    jCPm = Alpha * Beta * jCPm0

    # now evaluate final flux expressions you actually need
    out = {{}}
    {eval_lines}
    return out

def rhs(S, H, jCP, jSG, L, T, Nu, X, p):
    fx = compute_fluxes(S,H,jCP,jSG, L,T,Nu,X, p)
    # Example RHS; replace with your exact equations if different:
    dS   = S * (fx['jSG'] - fx['jST'])
    dH   = H * (fx.get('jHG', 0.0) - fx['jHT'])
    djCP = (fx['jCPm'] - jCP)   # if your rule gives aim directly, adapt naming
    djSG = (fx['jSGm'] - jSG)
    return dS, dH, djCP, djSG
'''

def main():
    resolved = expand_all(RULES, ALLOWED)
    # Filter: only keep the names you will access in compute_fluxes
    # e.g., those appearing in RHS: jSG, jST, jHG, jHT, jCPm, jSGm, etc.
    needed = ['jSG','jST','jHG','jHT','jCPm','jSGm']  # <-- adjust to your model
    block_lines = []
    for k in needed:
        if k not in resolved:
            print(f"[WARN] missing resolved expr for {k}")
            continue
        block_lines.append(f"{k} = '''{resolved[k]}'''")
    resolved_block = "\n".join(block_lines)

    # Binds for parameters used in TEMPLATE_BODY
    binds = []
    for name in sorted(PARAMS):
        binds.append(f"{name} = float(p['{name}'])")
    param_binds = "\n    ".join(binds)

    # Lines to eval expressions for needed fluxes
    eval_lines = []
    for name in needed:
        eval_lines.append(f"out['{name}'] = eval({name}, globals(), locals())")
    eval_code = "\n    ".join(eval_lines)

    code = TEMPLATE_HEADER + TEMPLATE_BODY.format(
        resolved_block=resolved_block,
        param_binds=param_binds,
        eval_lines=eval_code
    )

    Path('generated_rules.py').write_text(code)
    print("Wrote generated_rules.py")

if __name__ == '__main__':
    main()
