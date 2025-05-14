#!/usr/bin/env python3
# pipeline.py
"""
Master pipeline: run all five steps in order.
"""
from gen       import run_gen
from spec      import run_spec
from abundance import run_abundance
from assign    import run_assign
from damping   import run_damping

def main():
    print("→ Running gen.py …")
    run_gen()
    print("→ Running spec.py …")
    run_spec()
    print("→ Running abundance.py …")
    run_abundance()
    print("→ Running assign.py …")
    run_assign()
    print("→ Running damping.py …")
    run_damping()
    print("✅  All done!")

if __name__ == "__main__":
    main()
