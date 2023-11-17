import ipdb

ipdb.set_trace = lambda: 1

import argparse
import logging

import timm

from dev_tools import testing
from dev_tools import model_loading
from dev_tools import testing_result_registry as trr

BLACKLIST = ['efficientnet_l2']

def filter_blacklist(models: list):
    result = []
    for m in models:
        if m not in BLACKLIST:
            result.append(m)
    return result



def set_verbosity(level):
    mute = []
    if level == 0:
        mute.append("dev_tools")
    if level <= 1:
        mute.append("torch_dag")
    if level <= 2:
        mute.append("timm")
    for m in mute:
        logging.getLogger(m).setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser()
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--nas", type=str, default=None)
    model_group.add_argument("--timm", type=str, default=None)
    model_group.add_argument("--timm_detailed", type=str, default=None)
    model_group.add_argument("--timm_missing", action="store_true")
    parser.add_argument("-v", "--verbosity", type=int, default=1)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--skip_block_pruning_tests", action="store_true")
    parser.add_argument("--prob_orbit_removal", default=0.1, type=float)
    return parser.parse_args()


# def run_model_tests()


def main():
    args = parse_args()
    set_verbosity(args.verbosity)
    timm_names = args.timm or args.timm_detailed
    if timm_names is not None:
        if "<" in timm_names or ">" in timm_names:
            bounds = timm_names.split(",")
            matched_models = timm.list_models("*", pretrained=True)
            for b in bounds:
                sgn, name = b[0], b[1:]
                if sgn == "<":
                    if name[0] == "=":
                        name = name[1:]
                        matched_models = [m for m in matched_models if m <= name]
                    else:
                        matched_models = [m for m in matched_models if m < name]
                elif sgn == ">":
                    if name[0] == "=":
                        name = name[1:]
                        matched_models = [m for m in matched_models if m >= name]
                    else:
                        matched_models = [m for m in matched_models if m > name]
        else:
            matched_models = timm.list_models(timm_names, pretrained=False)
        if not args.timm_detailed:
            unique_prefixes = set(name.split(".")[0] for name in matched_models)
            unique_matched_models = []
            for up in unique_prefixes:
                first_match = [m for m in matched_models if up in m][0]
                unique_matched_models.append(first_match)
            matched_models = unique_matched_models
        matched_models = filter_blacklist(matched_models)
        model_loaders = [
            model_loading.TimmModelLoader(model_name) for model_name in matched_models
        ]
    elif args.timm_missing:
        all_models = filter_blacklist(timm.list_models())
        rr = trr.JsonResultRegistry.default()
        reg = rr.read()
        tested_models = set([name.split(".")[0] for name in list(reg.keys())])
        matched_models = [m for m in all_models if m not in tested_models and "enormous" not in m]
        model_loaders = [
            model_loading.TimmModelLoader(model_name) for model_name in matched_models
        ]
    else:
        model_loaders = [model_loading.NasModelLoader(args.nas)]
    if args.skip_block_pruning_tests:
        tests = [
            testing.WrappingTester(),
            testing.ChannelPruningTester(args.prob_orbit_removal),
        ]
    else:
        tests = testing.get_tests(channel_prob_removal=args.prob_orbit_removal)
    if args.save:
        rr = trr.JsonResultRegistry.default()
        if args.reset:
            rr.make_empty()
    else:
        if args.reset:
            raise Exception("--reset cannot be used without --save")
        rr = None
    results = testing.test_models(
        model_loaders,
        tests,
        result_registry=rr,
    )
    print(results)


if __name__ == "__main__":
    main()
