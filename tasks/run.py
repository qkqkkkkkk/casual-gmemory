import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import shutil
import yaml
from dataclasses import dataclass, field
import argparse
import random
from tqdm import tqdm

import mas
from mas.agents import Agent
from mas.module_map import module_map
from mas.reasoning import ReasoningBase
from mas.memory import MASMemoryBase
from mas.llm import LLMCallable, GPTChat, get_price
from mas.mas import MetaMAS
from mas.utils import EmbeddingFunc

from envs import BaseEnv, BaseRecorder, get_env, get_recorder, get_task
from mas_workflow import get_mas
from prompts import get_dataset_system_prompt, get_task_few_shots
from utils import get_model_type
from causal_memory_control import GMemoryExposureGate


with open('tasks/configs.yaml') as reader:
    CONFIG: dict = yaml.safe_load(reader)

WORKING_DIR: str = None

@dataclass
class TaskManager:
    task_name: str              # task name
    mas_type: str               # type of mas
    memory_type: str            # memory type
    tasks: list[dict]           # all tasks
    env: BaseEnv                # interative datatset environment
    recorder: BaseRecorder      # record experiment results
    mas: MetaMAS                # multi-agent system
    mas_config: dict = field(default_factory=dict)   # mas configs
    mem_config: dict = field(default_factory=dict)   # memory configs
    memory_gate: object = None


def build_task(task: str, mas_type: str, memory_type: str, max_steps: int) -> TaskManager:

    with open(CONFIG.get(task).get('env_config_path')) as reader:
        config = yaml.safe_load(reader)

    env: BaseEnv = get_env(task, config, max_steps)
    recorder: BaseRecorder = get_recorder(task, working_dir=WORKING_DIR, namespace='total_task')
    tasks: list[dict] = get_task(task)
    mas_workflow: MetaMAS = get_mas(mas_type)
    mas_config: dict = CONFIG.get(mas_type, {})

    return TaskManager(
        task_name=task,
        mas_type=mas_type,
        memory_type=memory_type,
        tasks=tasks,
        env=env,
        recorder=recorder,
        mas=mas_workflow,
        mas_config=mas_config
    )   

def build_mas(
    task_manager: TaskManager,
    reasoning: str = None,
    mas_memory: str = None,
    llm_type: str = None,
    memory_gate = None,
) -> None:
    
    embed_func = EmbeddingFunc(CONFIG.get('embedding_model', "sentence-transformers/all-MiniLM-L6-v2")) 
    reasoning_module_type, mas_memory_module_type = module_map(reasoning, mas_memory)

    llm_model: LLMCallable = GPTChat(model_name=llm_type)
    reasoning_module: ReasoningBase = reasoning_module_type(llm_model=llm_model)
    mas_memory_module: MASMemoryBase = mas_memory_module_type(
        namespace=mas_memory,
        global_config=task_manager.mem_config,
        llm_model=llm_model,
        embedding_func=embed_func 
    )
    
    task_manager.mas.add_observer(task_manager.recorder)  
    task_manager.mas.build_system(reasoning_module, mas_memory_module, task_manager.env, task_manager.mas_config)
    if memory_gate is not None:
        exposure_ids = list(task_manager.mas.agents_team)
        if task_manager.mas_type == 'macnet':
            decision_node = getattr(task_manager.mas, '_decision_node', None)
            if decision_node is not None:
                exposure_ids.append(decision_node.id)
        memory_gate.set_exposure_agent_ids(exposure_ids)
        mas_memory_module.set_retrieval_gate(memory_gate)
        task_manager.memory_gate = memory_gate

def run_task(task_manager: TaskManager) -> None:

    task_manager.recorder.dataset_begin()
    
    task_ids = getattr(task_manager, 'task_ids', list(range(len(task_manager.tasks))))
    for task_id, task_config in tqdm(zip(task_ids, task_manager.tasks), total=len(task_manager.tasks), desc="Running Tasks"):
        task_manager.recorder.task_begin(task_id, task_config)  
        
        task_main, task_description = task_manager.mas.env.set_env(task_config)   
        few_shots: list[str] = get_task_few_shots(
            dataset=task_manager.task_name, 
            task_config=task_config,
            few_shots_num=CONFIG.get(task_manager.task_name).get('few_shots_num', 0)
        )
        task_config.update(task_main=task_main, task_description=task_description, few_shots=few_shots)

        if task_manager.memory_gate is not None:
            task_manager.memory_gate.begin_task(task_id, task_config)
           
        task_instruction: str = get_dataset_system_prompt(task_manager.task_name, task_config=task_config)
        for agent in task_manager.mas.agents_team.values():    
            task_manager.recorder.log(f'------------ MAS Agent: {agent.name} ------------')
            task_manager.recorder.log(agent.add_task_instruction(task_instruction))

        reward, done = task_manager.mas.schedule(task_config) 

        if task_manager.memory_gate is not None:
            task_manager.memory_gate.end_task(reward, done)
    
        task_manager.recorder.task_end(reward, done)             
    
    task_manager.recorder.dataset_end()


def parse_task_ids(value: str) -> list[int]:
    selected = set()
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = (int(item) for item in part.split('-', 1))
            if end < start:
                raise argparse.ArgumentTypeError('task-id ranges must be ascending')
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    if not selected:
        raise argparse.ArgumentTypeError('at least one task id is required')
    return sorted(selected)


def select_tasks(task_manager: TaskManager, task_ids: list[int] | None, task_limit: int | None) -> None:
    available = len(task_manager.tasks)
    selected = task_ids if task_ids is not None else list(range(available))
    invalid = [task_id for task_id in selected if task_id < 0 or task_id >= available]
    if invalid:
        raise ValueError(f'task IDs out of range for dataset of size {available}: {invalid}')
    if task_limit is not None:
        if task_limit < 1:
            raise ValueError('task_limit must be positive')
        selected = selected[:task_limit]
    task_manager.task_ids = selected
    task_manager.tasks = [task_manager.tasks[task_id] for task_id in selected]


def build_memory_gate(args, working_dir: str):
    if args.memory_gate == 'off':
        return None
    candidate_kinds = tuple(
        value.strip() for value in args.gate_candidate_kinds.split(',') if value.strip()
    )
    common = dict(
        mode=args.memory_gate,
        delta=args.gate_delta,
        kappa=args.gate_kappa,
        candidate_kinds=candidate_kinds,
        max_drops=None if args.gate_max_drops < 0 else args.gate_max_drops,
        task_metadata={
            'task_type': args.task,
            'mas_type': args.mas_type,
            'model': args.model,
            'seed': args.seed,
            'memory_dir': os.path.abspath(args.memory_dir),
            'successful_topk': args.successful_topk,
            'failed_topk': args.failed_topk,
            'insights_topk': args.insights_topk,
            'retrieval_threshold': args.threshold,
            'hop': args.hop,
            'use_projector': args.use_projector,
        },
        log_path=args.gate_log or os.path.join(working_dir, 'memory_gate.jsonl'),
        resume=args.gate_resume,
    )
    if args.memory_gate == 'learned':
        return GMemoryExposureGate.from_checkpoint(args.gate_checkpoint, **common)
    return GMemoryExposureGate(**common)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tasks with specified modules.')
    parser.add_argument('--task', type=str, choices=['alfworld', 'fever', 'pddl'])
    parser.add_argument('--mas_type', type=str, choices=['autogen', 'macnet', 'dylan'])
    parser.add_argument('--mas_memory', type=str, default='none', help='Specify mas memory module')
    parser.add_argument('--reasoning', type=str, default='io', help='Specify reasoning module')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125', help='Specify the LLM model type')
    parser.add_argument('--max_trials', type=int, default=50, help='max number of steps')
    parser.add_argument('--successful_topk', type=int, default=1, help='Number of successful trajs to be retrieved from memory.')
    parser.add_argument('--failed_topk', type=int, default=0, help='Number of failed trajs to be retrieved from memory.')
    parser.add_argument('--insights_topk', type=int, default=3, help='Number of insights to be retrieved from memory.')
    parser.add_argument('--threshold', type=float, default=0.0, help='threshold for traj similarity.')
    parser.add_argument('--use_projector', action='store_true', help='whether to use role projector.')
    parser.add_argument('--hop', type=int, default=1, help='hop for traj similarity.')
    parser.add_argument('--seed', type=int, default=42, help='Python-side experiment seed.')
    parser.add_argument('--task_ids', type=parse_task_ids, default=None, help='Task IDs/ranges, e.g. 10-29,35.')
    parser.add_argument('--task_limit', type=int, default=None, help='Optional cap after --task_ids filtering.')
    parser.add_argument('--run_dir', type=str, default=None, help='Directory for logs/results; does not select the memory snapshot.')
    parser.add_argument('--memory_dir', type=str, default=None, help='Exact frozen g-memory persistence directory for gate evaluation.')
    parser.add_argument(
        '--memory_gate',
        choices=['off', 'always_keep', 'always_drop', 'learned'],
        default='off',
        help='Post-retrieval team exposure policy.',
    )
    parser.add_argument('--gate_checkpoint', type=str, default=None, help='Checkpoint created by causal_memory_control.train_gate.')
    parser.add_argument('--gate_delta', type=float, default=0.0, help='Minimum harmful-utility margin.')
    parser.add_argument('--gate_kappa', type=float, default=1.96, help='Uncertainty multiplier for conservative DROP.')
    parser.add_argument('--gate_max_drops', type=int, default=1, help='Max learned DROPs per retrieval; -1 means unlimited.')
    parser.add_argument('--gate_candidate_kinds', default='trajectory,insight', help='Comma-separated gate scope.')
    parser.add_argument('--gate_log', type=str, default=None, help='JSONL path for decisions and final team outcomes.')
    parser.add_argument('--gate_resume', action='store_true', help='Skip task IDs already completed in an existing gate log.')

    args = parser.parse_args()

    task: str = args.task
    mas_type: str = args.mas_type
    max_trials: int = args.max_trials
    model_type: str = args.model
    mas_memory_type: str = args.mas_memory
    reasoning_type: str = args.reasoning
    random.seed(args.seed)

    if args.memory_gate != 'off':
        if mas_memory_type != 'g-memory':
            parser.error('--memory_gate currently requires --mas_memory g-memory')
        if args.memory_dir is None:
            parser.error('gate experiments require --memory_dir so every arm uses one frozen snapshot')
        if not os.path.isdir(args.memory_dir):
            parser.error(f'--memory_dir does not exist: {args.memory_dir}')
        if args.memory_gate == 'learned' and args.gate_checkpoint is None:
            parser.error('--memory_gate learned requires --gate_checkpoint')
        if args.memory_gate == 'learned' and not os.path.isfile(args.gate_checkpoint):
            parser.error(f'--gate_checkpoint does not exist: {args.gate_checkpoint}')
    elif args.gate_resume:
        parser.error('--gate_resume requires an enabled --memory_gate')
    
    # dir
    default_run_dir = os.path.join(
        './.db',
        get_model_type(model_type),
        task,
        mas_type,
        f'{mas_memory_type}' if args.memory_gate == 'off' else f'{mas_memory_type}-{args.memory_gate}',
    )
    WORKING_DIR = args.run_dir or default_run_dir
    # if os.path.exists(WORKING_DIR):
    #     shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)
    
    # run tasks
    task_configs: TaskManager = build_task(task, mas_type, mas_memory_type, max_trials)
    select_tasks(task_configs, args.task_ids, args.task_limit)
    task_configs.mas_config['successful_topk'] = args.successful_topk
    task_configs.mas_config['failed_topk'] = args.failed_topk
    task_configs.mas_config['insights_topk'] = args.insights_topk
    task_configs.mas_config['threshold'] = args.threshold
    task_configs.mas_config['use_projector'] = args.use_projector
    task_configs.mem_config.update(
        working_dir=WORKING_DIR,
        hop=args.hop,
        persist_dir=args.memory_dir,
        read_only=args.memory_gate != 'off',
    )

    memory_gate = build_memory_gate(args, WORKING_DIR)
    if memory_gate is not None and memory_gate.completed_task_ids:
        pending = [
            (task_id, task_config)
            for task_id, task_config in zip(task_configs.task_ids, task_configs.tasks)
            if not memory_gate.task_is_complete(task_id)
        ]
        task_configs.task_ids = [task_id for task_id, _ in pending]
        task_configs.tasks = [task_config for _, task_config in pending]
    build_mas(task_configs, reasoning_type, mas_memory_type, model_type, memory_gate)
    run_task(task_configs)

    # postprocess
    completion_tokens, prompt_tokens, _ = get_price()
    task_configs.recorder.log(f'completion_tokens:{completion_tokens}, prompt_tokens:{prompt_tokens}, price={completion_tokens*15/1000000+prompt_tokens*5/1000000}')
