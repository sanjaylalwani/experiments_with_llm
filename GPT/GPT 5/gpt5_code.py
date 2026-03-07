import os
import tempfile
from typing import List, Optional

from openai import OpenAI
import mlflow
from dotenv import load_dotenv


# Load environment variables from .env in the repository root (if present)
load_dotenv()


class GPT5Client:
    """Wrapper around OpenAI `OpenAI` client that logs prompts and outputs to MLflow.

    Usage:
    - Provide `openai_api_key` or set `OPENAI_API_KEY` in env.
    - Optionally provide `mlflow_tracking_uri` and `mlflow_experiment_name`.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: Optional[str] = None,
        mlflow_enabled: bool = True,
    ) -> None:
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

        self.client = OpenAI()

        self.mlflow_enabled = mlflow_enabled
        if self.mlflow_enabled:
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
            if mlflow_experiment_name:
                mlflow.set_experiment(mlflow_experiment_name)

    def _log_to_mlflow(self, prompt: str, output: str, model: str) -> None:
        if not self.mlflow_enabled:
            return

        try:
            with mlflow.start_run():
                mlflow.log_param("model", model)
                # small previews as params
                mlflow.log_param("prompt_preview", prompt[:1000])
                mlflow.log_param("output_preview", output[:1000])

                # persist full prompt and output as artifacts
                with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as pf:
                    pf.write(prompt)
                    prompt_path = pf.name

                with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as of:
                    of.write(output)
                    output_path = of.name

                mlflow.log_artifact(prompt_path, artifact_path="prompts")
                mlflow.log_artifact(output_path, artifact_path="outputs")

                try:
                    os.remove(prompt_path)
                except Exception:
                    pass
                try:
                    os.remove(output_path)
                except Exception:
                    pass
        except Exception:
            # Logging must not break generation
            pass

    def generate(self, prompt: str, model: str = "gpt-5.4") -> str:
        """Generate text using the OpenAI client and log to MLflow.

        Returns the `output_text` attribute of the response when present,
        otherwise returns the stringified response.
        """
        response = self.client.responses.create(model=model, input=prompt)
        output = getattr(response, "output_text", None) or str(response)

        # log prompt/output to mlflow (non-blocking)
        self._log_to_mlflow(prompt=prompt, output=output, model=model)

        return output


if __name__ == "__main__":
    # Example usage
    # Ensure OPENAI_API_KEY is set in env or pass openai_api_key to constructor
    client = GPT5Client()
    print(client.generate("Write a short bedtime story about a unicorn."))
