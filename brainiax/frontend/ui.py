import itertools
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import gradio as gr
from fastapi import FastAPI
from gradio.themes.utils.colors import slate
from injector import inject, singleton
from llama_index.llms import ChatMessage, ChatResponse, MessageRole
from pydantic import BaseModel

from brainiax.constants import PROJECT_ROOT_PATH
from brainiax.di import global_injector
from brainiax.settings.settings import settings

logger = logging.getLogger(__name__)
THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)
AVATAR_BOT = THIS_DIRECTORY_RELATIVE / "avatar-bot.ico"

UI_TAB_TITLE = "BRAINIAX : Academic Aid AI Chatbot"

SOURCES_SEPARATOR = "\n\n Sources: \n"


class Source(BaseModel):
    file: str
    page: str
    text: str

    class Config:
        frozen = True

    @staticmethod
    def curate_sources(sources: list[None]) -> set["Source"]:
        curated_sources = set()

        for chunk in sources:
            doc_metadata = chunk.document.doc_metadata

            file_name = doc_metadata.get("file_name", "-") if doc_metadata else "-"
            page_label = doc_metadata.get("page_label", "-") if doc_metadata else "-"

            source = Source(file=file_name, page=page_label, text=chunk.text)
            curated_sources.add(source)

        return curated_sources


@singleton
class PrivateGptUi:
    @inject
    def __init__(
        self,
        ingest_service: None,
        chat_service: None,
        chunks_service: None,
    ) -> None:
        self._ingest_service = ingest_service
        self._chat_service = chat_service
        self._chunks_service = chunks_service

        # Cache the UI blocks
        self._ui_block = None

    # def _chat(self, message: str, history: list[list[str]], mode: str, *_: Any) -> Any:
    #     def yield_deltas(completion_gen: None) -> Iterable[str]:
    #         full_response: str = ""
    #         stream = completion_gen.response
    #         for delta in stream:
    #             if isinstance(delta, str):
    #                 full_response += str(delta)
    #             elif isinstance(delta, ChatResponse):
    #                 full_response += delta.delta or ""
    #             yield full_response

    #         if completion_gen.sources:
    #             full_response += SOURCES_SEPARATOR
    #             cur_sources = Source.curate_sources(completion_gen.sources)
    #             sources_text = "\n\n\n".join(
    #                 f"{index}. {source.file} (page {source.page})"
    #                 for index, source in enumerate(cur_sources, start=1)
    #             )
    #             full_response += sources_text
    #         yield full_response

    #     def build_history() -> list[ChatMessage]:
    #         history_messages: list[ChatMessage] = list(
    #             itertools.chain(
    #                 *[
    #                     [
    #                         ChatMessage(content=interaction[0], role=MessageRole.USER),
    #                         ChatMessage(
    #                             # Remove from history content the Sources information
    #                             content=interaction[1].split(SOURCES_SEPARATOR)[0],
    #                             role=MessageRole.ASSISTANT,
    #                         ),
    #                     ]
    #                     for interaction in history
    #                 ]
    #             )
    #         )

    #         # max 20 messages to try to avoid context overflow
    #         return history_messages[:20]

    #     new_message = ChatMessage(content=message, role=MessageRole.USER)
    #     all_messages = [*build_history(), new_message]
    #     match mode:
    #         case "Query Docs":
    #             # Add a system message to force the behaviour of the LLM
    #             # to answer only questions about the provided context.
    #             all_messages.insert(
    #                 0,
    #                 ChatMessage(
    #                     content="You can only answer questions about the provided context. If you know the answer "
    #                     "but it is not based in the provided context, don't provide the answer, just state "
    #                     "the answer is not in the context provided.",
    #                     role=MessageRole.SYSTEM,
    #                 ),
    #             )
    #             query_stream = self._chat_service.stream_chat(
    #                 messages=all_messages,
    #                 use_context=True,
    #             )
    #             yield from yield_deltas(query_stream)

    #         case "LLM Chat":
    #             llm_stream = self._chat_service.stream_chat(
    #                 messages=all_messages,
    #                 use_context=False,
    #             )
    #             yield from yield_deltas(llm_stream)

    #         case "Search in Docs":
    #             response = self._chunks_service.retrieve_relevant(
    #                 text=message, limit=4, prev_next_chunks=0
    #             )

    #             sources = Source.curate_sources(response)

    #             yield "\n\n\n".join(
    #                 f"{index}. **{source.file} "
    #                 f"(page {source.page})**\n "
    #                 f"{source.text}"
    #                 for index, source in enumerate(sources, start=1)
    #             )
    
    def _chat(self, message: str, history: list[list[str]], mode: str, *_: Any) -> Any:
        def yield_deltas() -> Iterable[str]:
            yield f"hello {message}"

        def build_history() -> list[ChatMessage]:
            history_messages: list[ChatMessage] = list(
                itertools.chain(
                    *[
                        [
                            ChatMessage(content=interaction[0], role=MessageRole.USER),
                            ChatMessage(content=interaction[1], role=MessageRole.ASSISTANT),
                        ]
                        for interaction in history
                    ]
                )
            )

            # max 20 messages to try to avoid context overflow
            return history_messages[:20]

        new_message = ChatMessage(content=message, role=MessageRole.USER)
        all_messages = [*build_history(), new_message]
        yield from yield_deltas()

    def upload_callback(self, file_contents_list, file_info_list):
        # Extract file names and update the list
        file_names = [file_info["name"] for file_info in file_info_list]
        self.uploaded_files = file_names

        # Update the ingested_dataset with the contents of the list
        self.ingested_dataset.set_data(self.uploaded_files)
    
    def _list_ingested_files(self) -> list[list[str]]:
        return [[row] for row in self.uploaded_files]

    def _build_ui_blocks(self) -> gr.Blocks:
        logger.debug("Creating the UI blocks")
        with gr.Blocks(
            title=UI_TAB_TITLE,
            theme=gr.themes.Soft(primary_hue=slate),
            css="body { "
                "margin: 0;"
                "padding: 0;"
                "display: flex;"
                "justify-content: center;"
                "align-items: center;"
                "height: 100vh;"
                "background-color: #000000;"  # Set your desired background color
                "}"
                ".logo { "
                "display:flex;"
                "background-color: #C7BAFF;"
                "height: 80px;"
                "border-radius: 8px;"
                "align-content: center;"
                "justify-content: center;"
                "align-items: center;"
                "}"
                "footer {display:none !important}"
                ".logo img { height: 25% }",
        ) as blocks:
            with gr.Row():
                gr.HTML(f"<div class='logo' style='font-weight: bold; color: black;font-size: 20px;'>BRAINIAX : Academic Aid AI Chatbot</div>")

            with gr.Row():
                with gr.Column(scale=3, variant="compact"):
                    mode = gr.Radio(
                        ["Query Docs", "Search in Docs", "LLM Chat"],
                        label="Mode",
                        value="Query Docs",
                    )
                    upload_button = gr.components.UploadButton(
                        "Upload File(s)",
                        type="filepath",
                        file_count="multiple",
                        size="sm",
                    )

                    ingested_dataset = gr.List(
                        #self._list_ingested_files,
                        headers=["File name"],
                        label="Ingested Files",
                        interactive=False,
                        render=False,  # Rendered under the button
                    )
                    upload_button.upload(
                        self.upload_callback,
                        inputs=upload_button,
                        outputs=ingested_dataset,
                    )
                    ingested_dataset.change(
                        self._list_ingested_files,
                        outputs=ingested_dataset,
                    )
                    ingested_dataset.render()
                with gr.Column(scale=7):
                    _ = gr.ChatInterface(
                        self._chat,
                        chatbot=gr.Chatbot(
                            label=f"LLM: {settings().llm.mode}",
                            show_copy_button=True,
                            render=False,
                            avatar_images=(
                                None,
                                AVATAR_BOT,
                            ),
                        ),
                        additional_inputs=[mode, upload_button],
                    )
        return blocks

    def get_ui_blocks(self) -> gr.Blocks:
        if self._ui_block is None:
            self._ui_block = self._build_ui_blocks()
        return self._ui_block

    def mount_in_app(self, app: FastAPI, path: str) -> None:
        blocks = self.get_ui_blocks()
        blocks.queue()
        logger.info("Mounting the gradio UI, at path=%s", path)
        gr.mount_gradio_app(app, blocks, path=path)


if __name__ == "__main__":
    ui = global_injector.get(PrivateGptUi)
    _blocks = ui.get_ui_blocks()
    _blocks.queue()
    _blocks.launch(debug=False, show_api=False)