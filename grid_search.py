import itertools
import os
import sys
import yaml

from ruamel.yaml import YAML
from ruamel.yaml.constructor import SafeConstructor

from utils.helper import set_logger, fill_gpu_jobs, get_tmp_yaml

def construct_yaml_map(self, node):
    # test if there are duplicate node keys
    data = []
    yield data
    for key_node, value_node in node.value:
        key = self.construct_object(key_node, deep=True)
        val = self.construct_object(value_node, deep=True)
        data.append((key, val))

def run():
    logger = set_logger()

    with open('grid.yaml') as fp:
        # settings = YAML().load(fp)
        settings = yaml.load(fp)
        logger.info('1'*100)
        logger.info('grid.yaml : ')
        logger.info(settings)
        logger.info('1'*100)
        test_set = sys.argv[1:] if len(sys.argv) > 1 else settings['common']['config']
        all_args = [settings[t] for t in test_set]
        entrypoint = settings['common']['entrypoint']
    with open('default.yaml') as fp:
        settings_default = yaml.load(fp)
        os.environ['suffix_model_id'] = settings_default['default']['suffix_model_id']

    cmd = ' '.join(['python app.py', entrypoint, '%s'])

    all_jobs = []
    for all_arg in all_args:
        k, v = zip(*[(k, v) for k, v in all_arg.items()])
        all_jobs += [{kk: pp for kk, pp in zip(k, p)} for p in itertools.product(*v)]
    while all_jobs:
        all_jobs = fill_gpu_jobs(all_jobs, logger,
                                 job_parser=lambda x: cmd % get_tmp_yaml(x,
                                                                         (os.environ['suffix_model_id'] if
                                                                          os.environ['suffix_model_id'] else
                                                                          '+'.join(test_set)) + '-'),
                                 wait_until_next=settings['common']['wait_until_next'],
                                 retry_delay=settings['common']['retry_delay'],
                                 do_shuffle=True)

    logger.info('all jobs are done!')


if __name__ == '__main__':
    run()
