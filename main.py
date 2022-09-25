from utils import *
from scheme import *


def main():
    backup()
    print_args()
    init_state()
    model = build_model()
    scenario, scenario_eval = load_scenario()
    Scheme = eval(args.scheme)
    for taskid, traindata in enumerate(scenario, start=1):
        scheme = Scheme(model, traindata, taskid)
        model = scheme.train()
        evaluate(model, scenario_eval, taskid)
        if args.end is not None and taskid == args.end:
            break


if __name__ == '__main__':
    main()
