# Model Serving

Inferencing in SEA-HELM is supported through the use of the vLLM and LiteLLM inference frameworks.
The following model types are accepted: `vllm`, `litellm`, `openai`, `none`

## vLLM

The `VLLMServing` class serves the model using the offline inference method found in vLLM. This allows for any model that is supported by vLLM to be served. Additionally, vLLM engine arguments can be configured using the `--model_args` cli argument. For the full list of engine args, please see the vLLM documentation on [Engine Args](https://docs.vllm.ai/en/latest/serving/engine_args.html#engine-args)

## LiteLLM

The `LiteLLMServing` class interfaces with the liteLLM package to provide support for closed source API servers such as OpenAI, Claude and Vertex.

> [!Important]  
> **Specifying the model provider**  
> Please ensure that the model provider is specified using the `api_provider` in `--model_args`:
>
> - Example (OpenAI): `api_provider=openai`
> - Example (Anthropic): `api_provider=anthropic`

It also supports the use of vLLM OpenAI-Compatible Server that is started using `vllm serve`. Please ensure that the correct `api_provider`, `base_url` and `api_key` are passed as one of the model_args. For example:

```bash
--model_args api_provider=openai,base_url=http://localhost:8000/v1,api_key=token-abc123
```

> [!Tip]  
> **Tokenization of prompts**  
> The evaluation framework will make an additional call to tokenize the prompts so as to gather statistics on the given prompt. If there are no tokenization end points available, please set the flag `--skip_tokenize_prompts`.

> [!Tip]  
> **Setting SSL verify to `False`**  
> To set SSL verify to false, please pass the key `ssl_verify=False` as one of the `--model_args`

## OpenAI (Batching API)

Support for the OpenAI Batch API is also provided. This provided a cost saving at the expense of potentially having to wait for longer if the OpenAI server are busy. To run this set:

```bash
--model_type openai
--model_args "api_provider=openai"
```

## None

Setting `model_type` to `none` is a special case to allow for the recalculation of evaluation metrics without any new inference being made. As such, no model will be loaded for vLLM and no API calls will be made. Please ensure that all the results are cached in the inference folder before running this.
