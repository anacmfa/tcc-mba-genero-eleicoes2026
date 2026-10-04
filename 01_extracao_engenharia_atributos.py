"""
Etapa 2: Adaptação de Domínio (MLM) via BERTimbau,
Ajuste Fino Supervisionado (HITL) e Cálculo das Métricas de Classificação
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ------------------------------------------------------------------------------
# 1. FUNÇÃO DE AVALIAÇÃO DE MÉTRICAS (Precisão, Revocação, Acurácia, Macro F1)
# ------------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "acuracia": acc,
        "precisao": precision,
        "recall": recall,
        "macro_f1": f1,
    }


# ------------------------------------------------------------------------------
# 2. FASE 1: ADAPTAÇÃO DE DOMÍNIO (MLM) NO CORPUS ELEITORAL
# ------------------------------------------------------------------------------
def treinar_adaptacao_dominio(dataset_nacional_tokenizado):
    """Realiza a continuidade do pré-treinamento com modelagem de linguagem mascarada."""
    modelo_base = "neuralmind/bert-base-portuguese-cased"
    tokenizer = AutoTokenizer.from_pretrained(modelo_base)
    model_mlm = AutoModelForMaskedLM.from_pretrained(modelo_base)

    # Mascaramento estocástico de 15% dos tokens
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    args_mlm = TrainingArguments(
        output_dir="./bert_adaptado",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        fp16=torch.cuda.is_available(),
        save_strategy="no",
        logging_steps=50,
    )

    trainer_mlm = Trainer(
        model=model_mlm,
        args=args_mlm,
        data_collator=data_collator,
        train_dataset=dataset_nacional_tokenizado,
    )

    trainer_mlm.train()
    trainer_mlm.save_model("./bert2_dominio_eleitoral")
    tokenizer.save_pretrained("./bert2_dominio_eleitoral")
    print("Adaptação de domínio concluída e pesos guardados em ./bert2_dominio_eleitoral")


# ------------------------------------------------------------------------------
# 3. FASE 2: AJUSTE FINO SUPERVISIONADO (PADRÃO-OURO HITL)
# ------------------------------------------------------------------------------
def treinar_classificador_hitl(dataset_treino_hitl, dataset_val_hitl):
    """Executa o ajuste fino supervisionado do BERT2 para classificação binária."""
    caminho_modelo_adaptado = "./bert2_dominio_eleitoral"

    model_clf = AutoModelForSequenceClassification.from_pretrained(
        caminho_modelo_adaptado, num_labels=2
    )

    args_clf = TrainingArguments(
        output_dir="./bert2_classificador_genero",
        num_train_epochs=5,
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),
    )

    trainer_clf = Trainer(
        model=model_clf,
        args=args_clf,
        train_dataset=dataset_treino_hitl,  # 80% do conjunto anotado manualmente
        eval_dataset=dataset_val_hitl,      # 20% reservado para validação
        compute_metrics=compute_metrics,
    )

    trainer_clf.train()
    trainer_clf.save_model("./bert2_classificador_genero_final")
    print("Treino do classificador concluído. Modelo pronto para inferência.")


if __name__ == "__main__":
    print("Pipeline de modelagem BERT2 pronto para execução.")
