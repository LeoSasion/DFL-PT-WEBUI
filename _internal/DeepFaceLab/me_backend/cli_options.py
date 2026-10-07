"""Argument definitions shared by CLI entry points; no neural runtime imports."""
import argparse


def config_options(parser, required=False):
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument('--config', help='JSON configuration file; partial updates merge with a resumed checkpoint')
    group.add_argument('--config-json', help='Inline JSON configuration object; partial updates merge with a resumed checkpoint')


def training_options(parser):
    config_options(parser)
    parser.add_argument('--resolution', type=int, default=None, help='New models default to 128; explicit resume changes are checked')
    parser.add_argument('--batch-size', type=int, default=None, help='New models default to 4; explicit resume changes require permission flag')
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--preview-every', type=int, default=10)
    parser.add_argument('--backup-every', type=int, default=1000, help='Automatic verified backup interval in iterations')
    parser.add_argument('--backup-keep', type=int, default=3, help='Automatic backup generations to retain (2..32)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reset-data-state', action='store_true')
    parser.add_argument('--allow-config-change', action='store_true', help='Explicitly apply compatible training settings during resume')
    parser.add_argument('--reset-optimizer', action='store_true', help='Explicitly discard optimizer history when changing optimizer type')
    parser.add_argument('--initialize-from', help='Initialize only network weights from a native pretrained me.pt or model directory')
    parser.add_argument('--pretraining-data-dir', help='Dedicated aligned faceset required when pretrain=true')
    parser.add_argument('--target-iterations', type=int, default=0, help='Save and stop at this total iteration; zero disables the limit')
    parser.add_argument('--debug-samples', action='store_true')
    parser.add_argument('--archi', default=None)
    parser.add_argument('--rg', dest='use_rg', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--fp16', dest='use_fp16', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--optimizer-on-cpu', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--pretrain', action=argparse.BooleanOptionalAction, default=None)
