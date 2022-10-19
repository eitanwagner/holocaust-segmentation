#
def make_data():
    import json
    data_path="/cs/snapless/oabend/eitan.wagner/segmentation/data/"
    with open(data_path + 'sf_raw_text.json', 'r') as infile:
        data = json.load(infile)

    r = ["sf_43019", "sf_38929", "sf_32788", "sf_38936", "sf_20505", "sf_23579", "sf_48155", "sf_35869",
         "sf_30751", "sf_30753", "sf_45091", "sf_25639", "sf_46120", "sf_32809", "sf_34857", "sf_46122", "sf_30765",
         "sf_24622", "sf_21550", "sf_26672"]
    r = [int(_r[3:]) for _r in r]

    with open(data_path + 'sf_unused.json', 'r') as infile:
        unused = json.load(infile)

    for_train = set(unused) - set(r)

    for_train, for_eval = list(for_train)[:int(0.9*len(for_train))], list(for_train)[int(0.9*len(for_train)):]

    train_texts = [f"\n\n\n== Holocaust Testimony {f_t} ==\n\n" + data[str(f_t)] for f_t in for_train]
    train_text = "".join(train_texts)
    val_texts = [f"\n\n\n== Holocaust Testimony {f_t} ==\n\n" + data[str(f_t)] for f_t in for_eval]
    val_text = "".join(val_texts)

    with open(data_path + "train_text", "w") as outfile:
        outfile.write(train_text)

    with open(data_path + "val_text", "w") as outfile:
        outfile.write(val_text)


########
# /cs/snapless/oabend/eitan.wagner/py3-venv3/bin/python run_clm.py \
#     --model_name_or_path gpt2-large \
#     --train_file /cs/snapless/oabend/eitan.wagner/segmentation/data/train_text \
#     --validation_file /cs/snapless/oabend/eitan.wagner/segmentation/data/val_text \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --do_train \
#     --do_eval \
#     --output_dir /cs/snapless/oabend/eitan.wagner/segmentation/test-clm

def train():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2-large", cache_dir='/cs/snapless/oabend/eitan.wagner/cache/')

    train_path = '/cs/snapless/oabend/eitan.wagner/segmentation/data/train_text'
    test_path = '/cs/snapless/oabend/eitan.wagner/segmentation/data/val_text'

    from transformers import TextDataset, DataCollatorForLanguageModeling

    def load_dataset(train_path,test_path,tokenizer):
        train_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=train_path,
            block_size=128)

        test_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=test_path,
            block_size=128)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False,
        )
        return train_dataset, test_dataset, data_collator

    train_dataset, test_dataset, data_collator = load_dataset(train_path,test_path,tokenizer)

    from transformers import Trainer, TrainingArguments, AutoModelWithLMHead

    # model = GPT2LMHeadModel.from_pretrained('gpt2'+self.large, cache_dir='/cs/snapless/oabend/eitan.wagner/cache/')
    model = AutoModelWithLMHead.from_pretrained("gpt2-large", cache_dir='/cs/snapless/oabend/eitan.wagner/cache/')


    training_args = TrainingArguments(
        output_dir="/cs/snapless/oabend/eitan.wagner/segmentation/models/gpt2-2", #The output directory
        overwrite_output_dir=True, #overwrite the content of the output directory
        num_train_epochs=3, # number of training epochs
        gradient_accumulation_steps=8,
        per_device_train_batch_size=1,  # batch size for training
        learning_rate=5e-5,
        per_device_eval_batch_size=1,  # batch size for evaluation
        eval_steps=400, # Number of update steps between two evaluations.
        save_steps=5000, # after # steps model is saved
        warmup_steps=500,# number of warmup steps for learning rate scheduler
        prediction_loss_only=True,
        logging_steps=10,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

    trainer.save_model()

def perplexity():
    import json
    data_path = "/cs/snapless/oabend/eitan.wagner/segmentation/data/"
    model_path = "/cs/snapless/oabend/eitan.wagner/segmentation/models/"
    with open(data_path + 'sf_raw_text.json', 'r') as infile:
        data = json.load(infile)
    with open(data_path + 'sf_unused.json', 'r') as infile:
        unused = json.load(infile)

    test = "\n\n".join([text for t, text in data.items() if int(t) not in unused][:100])
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    device = "cuda"
    # models = ["gpt2-xl", "gpt2-large", "gpt2-2", "gpt2-1"]
    models = ["gpt2", "gpt2-2", "gpt2-1"]
    # models = ["gpt2-xl", "gpt2-large", "gpt2-1"]
    ppls = []
    for model_id in models:
        _model_path = model_path
        if model_id in ["gpt2-xl", "gpt2-large", "gpt2"]:
            _model_path = ""
        if model_id == "gpt2-1":
            _model_path = "/cs/labs/oabend/eitan.wagner/models/"
        model = GPT2LMHeadModel.from_pretrained(_model_path+model_id,
                                                cache_dir='/cs/snapless/oabend/eitan.wagner/cache/').to(device)
        if model_id != 'gpt2-xl':
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large', cache_dir='/cs/snapless/oabend/eitan.wagner/cache/')
        else:
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-xl', cache_dir='/cs/snapless/oabend/eitan.wagner/cache/')

        # from datasets import load_dataset

        # test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        encodings = tokenizer(test, return_tensors="pt")

        import torch
        from tqdm import tqdm

        max_length = model.config.n_positions
        stride = 512
        end_loc = 1

        nlls = []
        for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len

            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        print(f"perplexity: {ppl}")
        ppls.append(ppl)
    return ppls

if __name__ == "__main__":
    # train()
    print(perplexity())