import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class FlanT5:
    def __init__(self, model_name: str = "/home/jiso/Documents/EPITA/action-learning/rag_medical/src/pipelines/ml_modles/flant5"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only = True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only = True).to(self.device)

    def recommend(self, question:str, max_length: int = 64) -> str:
        # prompt = (
        #     "Generate exactly four medically relevant follow-up questions based on the patient’s input question. "
        #     "Each follow-up question must be concise, end with a question mark, and explore a different aspect of the topic "
        #     "(e.g., diagnosis, treatment, risk factors, prognosis, prevention, or causes). "
        #     "The follow-up questions must not repeat the patient’s question or use its exact wording. "
        #     "Format the output as a numbered list (e.g., '1. Question?\n2. Question?\n3. Question?\n4. Question?').\n\n"
        #     "Example 1:\n"
        #     "Patient: \"What are the symptoms of diabetes?\"\n"
        #     "Follow-Up Questions:\n"
        #     "1. How is diabetes diagnosed?\n"
        #     "2. What are the treatment options for diabetes?\n"
        #     "3. Who is at risk for developing diabetes?\n"
        #     "4. What complications can arise from diabetes?\n\n"
        #     "Example 2:\n"
        #     "Patient: \"What is breast cancer?\"\n"
        #     "Follow-Up Questions:\n"
        #     "1. What are the main risk factors for breast cancer?\n"
        #     "2. How is breast cancer diagnosed?\n"
        #     "3. What treatments are available for breast cancer?\n"
        #     "4. What is the prognosis for breast cancer patients?\n\n"
        #     "Input Question: \"{}\"\n"
        #     "Follow-Up Questions:".format(question)
        # )
        prompt = (
            "Given the following medical question, generate four medically relevant follow-up questions:\n\n"
            "Input Question: \"{}\"\n"
            "Follow-Up Questions:"
        ).format(question)

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        output = self.model.generate(
                                    input_ids,
                                    max_length=max_length,
                                    num_beams=6,
                                    temperature=0.7,
                                    top_p=0.9,
                                    no_repeat_ngram_size=3,
                                    repetition_penalty=1.2,
                                    early_stopping=True)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Parse the pipe-separated questions into a list
        # return [q.strip() for q in result.split('|') if q.strip()]

