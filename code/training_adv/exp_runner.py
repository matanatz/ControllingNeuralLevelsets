import argparse
import sys
sys.path.append('../code')
from training_adv.train_adv import TrainRunner
from training_adv.eval_adv import EvalRunner
import GPUtil


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--nepoch', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--conf', type=str,required=True)
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU 0]')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--threads', type=int, default=8, help='multi threaded')
    parser.add_argument('--parallel', default=False, action="store_true")
    parser.add_argument(
            "--debug",
            dest="debug",
            default=True,
            action="store_true",
            help="If set, debugging messages will be printed",
        )
    parser.add_argument(
            "--quiet",
            "-q",
            dest="quiet",
            default=False,
            action="store_true",
            help="If set, only warnings will be printed",
        )

    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--eval', default=False, action="store_true")
    parser.add_argument('--timestamp',type=str )
    parser.add_argument('--checkpoint', type=str,default='latest')
    parser.add_argument('--eval_frequency', type=int, default=10)


    opt = parser.parse_args()


    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
                                    excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu

    # TODO save parser argumernts
    if not opt.eval:
        runner = TrainRunner(conf=opt.conf,
                                  batch_size=opt.batch_size,
                                nepochs=opt.nepoch,
                                expname=opt.expname,
                                gpu_index=gpu,
                                exps_folder_name='exps',
                                  threads=opt.threads,
                                  debug=opt.debug,
                                  quiet=opt.quiet,
                                  parallel=opt.parallel,
                                  is_continue=opt.is_continue,
                                timestamp=opt.timestamp,
                                checkpoint=opt.checkpoint,
                                eval_frequency =opt.eval_frequency,
                                 only_eval = opt.eval)
    else:
        runner = EvalRunner(conf=opt.conf,
                                  batch_size=opt.batch_size,
                                expname=opt.expname,
                                gpu_index=gpu,
                                exps_folder_name='evaluation',
                                threads=opt.threads,
                                checkpoint=opt.checkpoint)


    runner.run()