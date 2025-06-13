import asyncio

import pandas as pd
from llama_index.core.indices.base import BaseIndex

from src.core_managers.vector_store_manager import VectorStoreManager
from src.services.chat_bot_service import ChatBotService
from src.utils.directory_manager import DirectoryManager
from src.utils.enums import ChatBotConfig, IngestionConfig
from src.utils.helpers import get_unique_id

vt_store = VectorStoreManager()

index_path = IngestionConfig.INDEX_PATH
index_cancer = vt_store.build_or_load_index(f"{index_path}/{ChatBotConfig.CANCER}")
index_diabetes = vt_store.build_or_load_index(f"{index_path}/{ChatBotConfig.DIABETES}")
index_genetic = vt_store.build_or_load_index(
    f"{index_path}/{ChatBotConfig.GENETIC_AND_RARE_DISEASES}"
)
index_hormone = vt_store.build_or_load_index(
    f"{index_path}/{ChatBotConfig.GROWTH_HORMONE_RECEPTOR}"
)
index_heart_lung_blood = vt_store.build_or_load_index(
    f"{index_path}/{ChatBotConfig.HEART_LUNG_AND_BLOOD}"
)
index_neuro_disorders_and_stroke = vt_store.build_or_load_index(
    f"{index_path}/{ChatBotConfig.NEUROLOGICAL_DISORDERS_AND_STROKE}"
)
index_senior_health = vt_store.build_or_load_index(
    f"{index_path}/{ChatBotConfig.SENIOR_HEALTH}"
)
index_others = vt_store.build_or_load_index(f"{index_path}/{ChatBotConfig.OTHERS}")

all_indices = {
    ChatBotConfig.CANCER: index_cancer,
    ChatBotConfig.DIABETES: index_diabetes,
    ChatBotConfig.GENETIC_AND_RARE_DISEASES: index_genetic,
    ChatBotConfig.GROWTH_HORMONE_RECEPTOR: index_hormone,
    ChatBotConfig.HEART_LUNG_AND_BLOOD: index_heart_lung_blood,
    ChatBotConfig.NEUROLOGICAL_DISORDERS_AND_STROKE: index_neuro_disorders_and_stroke,
    ChatBotConfig.SENIOR_HEALTH: index_senior_health,
    ChatBotConfig.OTHERS: index_others,
}


async def run_rag(user_id: str, query: str, index: BaseIndex, use_cot: bool) -> str:
    chat_bot_service = ChatBotService(
        user_id=user_id,
        index=index,
        force_use_tools=True,
        use_cot=use_cot,
    )
    nearest_nodes = await chat_bot_service.retrieve_related_nodes(message=query)
    return await chat_bot_service.asynthesize_response(
        message=query, nearest_nodes=nearest_nodes
    )


async def compare_rag(
    user_id: str,
    query: str,
    domain_name: str,
) -> (str, str):
    index = all_indices[domain_name]
    response_with_cot = await run_rag(user_id, query, index, use_cot=True)
    response_without_cot = await run_rag(user_id, query, index, use_cot=False)
    return response_with_cot, response_without_cot


async def main():
    df_query = pd.read_csv("src/data/questions_for_rag/questions.csv")
    log_file_path = "src/data/rag_comparison_log/logs.csv"
    DirectoryManager.create_dir_if_not_exists("src/rag_comparison_log")
    queries = df_query["question"].tolist()
    domains = df_query["domain"].tolist()
    for i in range(len(df_query)):
        user_id = str(get_unique_id())
        query = queries[i]
        domain = domains[i]
        print(f"Processing query: {query} for domain: {domain}")
        response_with_cot, response_without_cot = await compare_rag(
            user_id, query, domain
        )

        data = {
            "user_id": user_id,
            "query": query,
            "domain": domain,
            "response_with_cot": response_with_cot,
            "response_without_cot": response_without_cot,
        }

        if not DirectoryManager.check_if_file_exists(log_file_path):
            col_names = list(data.keys())
            DirectoryManager.create_empty_csv_file(
                col_names=col_names, file_path=log_file_path
            )

        DirectoryManager.write_log_file(
            log_file_path,
            data,
        )


if __name__ == "__main__":
    asyncio.run(main())
