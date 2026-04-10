import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from loguru import logger
import bitsandbytes as bnb
from dataclasses import dataclass
from peft import LoraConfig, get_peft_model
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from transformers import DataCollator, PreTrainedTokenizerBase, EvalPrediction, TrainerCallback
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, PreTrainedModel, AutoModelForCausalLM,BitsAndBytesConfig


class SFTDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length: int, template):
        self.tokenizer = tokenizer
        self.template_name = template.template_name
        self.system_format = template.system_format         # 系统提示 (Role)
        self.user_format  = template.user_format            # 用户提示 (Question)
        self.assistant_format = template.assistant_format   # 助手回复 (Answer)
        self.system = template.system

        self.max_seq_length = max_seq_length
        logger.info("Loading data: {}".format(file))
        with open(file,'r',encoding='utf8') as f:
            data_list = f.readlines()
        logger.info(f"Use template: {self.template_name} for training")
        logger.info("There are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        data = json.loads(data)
        input_ids, target_mask = [], []

        if self.system_format is not None:
            # 如果自己设定了系统提示词 否则使用默认的
            system = data['Instruction'].strip() if 'Instruction' in data.keys() else self.system
            if system is not None:
                system_text = self.system_format.format(content=system)
                input_ids = self.tokenizer.encode(system_text,add_special_tokens=False)
                target_mask = [0]*len(input_ids)
        
        human =  data['Text'] + '\n' + data['Target']
        assistant = data['Stance'].strip()

        human = self.user_format.format(content=human, stop_token=self.tokenizer.eos_token)
        assistant = self.assistant_format.format(content=assistant, stop_token=self.tokenizer.eos_token)

        input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
        output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

        input_ids += input_tokens + output_tokens
        target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)
        assert len(input_ids) == len(target_mask), "Input_ids_len != Target_mask_len "
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1]*len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }


class SFTDataCollator(object):
    def __init__(self,tokenizer,max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = self.tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str,Any]]) -> Dict[str,Any]:
        lengths = [len(x['input_ids']) for x in batch if x['input_ids'] is not None]
        batch_max_len = min(max(lengths),self.max_seq_length)
        input_ids_batch, attention_mask_batch, target_mask_batch = [], [], []
        for x in batch:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            target_mask = x['target_mask']

            if input_ids is None:
                logger.info('some input_ids is None')
                continue

            padding_len = batch_max_len - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len

            input_ids_batch.append(input_ids[:self.max_seq_length])
            attention_mask_batch.append(attention_mask[:self.max_seq_length])
            target_mask_batch.append(target_mask[:self.max_seq_length])

        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)

        labels = torch.where(target_mask_batch == 1, input_ids_batch, -100)
        inputs = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': labels
        }
        return inputs


class MyTrainer(Trainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None, # type: ignore
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
            compute_loss=None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.loss_func = compute_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch= None):
        if self.loss_func is None:
            loss = super().compute_loss(model, inputs, return_outputs)
        else:
            loss = self.loss_func(model, inputs, return_outputs)
        return loss

class Loss(object):
    """
    所有 Loss 的类父类
    """
    def __call__(self, model, inputs,return_outputs=False):
        """
        todo label smoothing
        用于计算loss。
        看源码发现，return_outputs=True为train时调用，return_outputs=False为eval和predict调用
        :param model: 模型
        :param inputs: 模型输入，dict
        :param return_outputs:是否返回模型的输出
        :return:
        """
        raise NotImplemented


class Target_Loss(Loss):
    def __init__(self, ignore_index):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_function = nn.CrossEntropyLoss(ignore_index=ignore_index)
    def __call__(self, model, inputs, return_outputs=False):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs['labels']
        outputs = model(input_ids=input_ids,attention_mask=attention_mask,return_dict=True)
        logits = outputs['logits'] if isinstance(outputs,dict) else outputs[0]
        shift_logits = logits[:,:-1,:].contiguous()
        shift_labels = labels[:,1:].contiguous()
        loss = self.loss_function(shift_logits.view(-1,shift_logits.size(-1)),shift_labels.view(-1))
        return (loss,outputs) if return_outputs else loss


@dataclass
class Template:
    template_name:str
    system_format: str
    user_format: str
    assistant_format: str
    system: str
    stop_word: str

template_dict: Dict[str, Template] = dict()

def register_template(template_name, system_format, user_format, assistant_format, system, stop_word=None):
    template_dict[template_name] = Template(
        template_name=template_name,
        system_format=system_format,
        user_format=user_format,
        assistant_format=assistant_format,
        system=system,
        stop_word=stop_word,
        # stop_token_id=stop_token_id
    )

register_template(
    template_name='Qwen',
    system_format='<|im_start|>system:{content}<|im_end|>\n',
    user_format='<|im_start|>user:{content}<|im_end|>\n<|im_start|>assistant:',
    assistant_format='{content}<|im_end|>',
    system="You as an expert in sentiment analysis within Natural Language Processing.",
    stop_word='<|im_end|>'
)


def find_all_linear_names(model,train_mode):
    assert train_mode in ['lora','qlora']
    lora_module_names = set()
    type = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear
    for name, module in model.named_modules():
        if isinstance(module,type):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    
    lora_module_names = list(lora_module_names)
    logger.info(f'{train_mode} target module names:{lora_module_names}')
    return lora_module_names


def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params = all_params + param.numel()
        if param.requires_grad:
            trainable_params = trainable_params + param.numel()
    logger.info(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.2f}")


def load_pretrain_dataset(args,tokenizer):
    if args.template_name not in template_dict.keys():
        raise Exception(f"template_name doesn't exist, all template_name: {template_dict.keys()}")
    template = template_dict[args.template_name]
    logger.info('Loading data with PretrainDataset')
    train_dataset = SFTDataset(args.train_file, tokenizer, args.max_seq_length, template)

    return train_dataset


if __name__ == "__main__":
    # 1.Instantiate a base model.
    class TrainArgs:
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        template_name = "Qwen"
        train_file = "my_data.jsonl"
        gradient_checkpointing = True
        max_seq_length = 1024
    
    args = TrainArgs
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True, 
        bnb_8bit_compute_dtype=torch.bfloat16,
    )

    model_kwags = dict(
        trust_remote_code=True,
        torch_dtype = torch.bfloat16,  # 保持和 TrainingArguments 中 bf=True 一致
        use_cache=False if args.gradient_checkpointing else True,
        # quantization_config=quantization_config,
        quantization_config=None
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name,**model_kwags)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2.Create a configuration (LoraConfig) where you define LoRA-specific parameters.
    peft_config = LoraConfig(
        r=8,                        # LoRA 中低秩矩阵的秩
        lora_alpha=32,              # 缩放因子
        lora_dropout=0.1,
        target_modules=find_all_linear_names(model, train_mode='lora'),
        task_type='CAUSAL_LM',      # 指定任务类型
    )

    # 3.Wrap the base model with get_peft_model() to get a trainable PeftModel. (冻结参数 添加 Adapter)
    model = get_peft_model(model, peft_config)
    logger.info(f'Memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
    print_trainable_parameters(model)

    # 4.Train the PeftModel as you normally would train the base model.
    training_args = TrainingArguments(
        output_dir=f"{args.model_name}-lora",
        learning_rate=2e-4,
        num_train_epochs=30,
        per_device_train_batch_size=1,
        save_total_limit=3,
        save_strategy="epoch",
        logging_steps=1000,
        remove_unused_columns=False,
        label_names=["labels"],
        bf16=True,
        # enable_thinking=False
    )

    loss_function = Target_Loss(ignore_index=-100)
    train_set = load_pretrain_dataset(args,tokenizer)
    data_collator = SFTDataCollator(tokenizer=tokenizer,max_seq_length=args.max_seq_length)
    trainer = MyTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_set,
        compute_loss=loss_function
        )
    trainer.train()