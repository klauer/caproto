#!/usr/bin/env python3
import ast

from caproto.server import PVGroup, pvproperty, run, template_arg_parser


def create_pv_group(pv_to_value_dict):
    body = {}
    for idx, (pv, value) in enumerate(pv_to_value_dict.items()):
        kwargs = {}
        if isinstance(value, str):
            kwargs['report_as_string'] = True
        body[f'attr_{idx}'] = pvproperty(
            name=pv,
            value=value,
            **kwargs
        )

    return type('BucketOPVs', (PVGroup,), body)


if __name__ == '__main__':
    parser, split_args = template_arg_parser(
        default_prefix='',
        desc='An IOC that serves a bucket of PVs.')

    parser.add_argument('--filename',
                        help='The file to read the PVs from',
                        required=True, type=str)
    args = parser.parse_args()
    ioc_options, run_options = split_args(args)

    def split_lines():
        with open(args.filename, 'r') as fp:
            pv_and_value = fp.read().splitlines()
        for line in pv_and_value:
            yield line.split(',', 1)

    pvs = {
        pv: ast.literal_eval(value)
        for pv, value in split_lines()
    }
    klass = create_pv_group(pvs)

    ioc = klass(**ioc_options)
    run(ioc.pvdb, **run_options)
