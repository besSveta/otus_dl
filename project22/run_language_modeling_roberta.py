from transformers import LineByLineTextDataset
from transformers import RobertaTokenizer
from transformers import RobertaForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers import RobertaConfig
from tokenizers import ByteLevelBPETokenizer

from transformers import DataCollatorForLanguageModeling


def main():
    tokenizer = ByteLevelBPETokenizer()
    path = "osetian_corpus6.txt"
    model_path = "./models/OssetBERTo-small6"
    # Customize training

    vocab_size = 70_000

    tokenizer.train(files=path, vocab_size=vocab_size, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])
    # Save files to disk
    tokenizer.save_model(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path, max_len=512)

    config = RobertaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    model = RobertaForMaskedLM(config=config)
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=path,
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(model_path)


if __name__ == "__main__":
    main()
