

# **Technical Analysis and Implementation Guide for the SGLang Observability Harness**

## **Executive Summary**

This report provides a comprehensive technical analysis and a set of actionable recommendations for the development of the SGLang observability container and its associated benchmarking harness. The analysis addresses six critical open questions concerning the SGLang API surface, the behavior of "thinking" models such as Qwen3-Next, and the practicalities of achieving reliable metric collection and reproducibility.

The primary conclusion of this investigation is that the /v1/chat/completions endpoint is the only stable, production-ready transport mechanism suitable for the proposed benchmarking harness. The alternative /v1/responses API, while present in the SGLang codebase, is functionally incomplete and exhibits critical bugs that render it unreliable for correctness-first testing. Consequently, the harness design should standardize exclusively on the Chat Completions API.

Key findings validate the proposed design principle of excluding model reasoning from conversational history during multi-turn tests. Official documentation from model providers like Qwen explicitly mandates this approach to ensure conversational coherence. Furthermore, the analysis reveals that true bit-for-bit determinism via the seed parameter is not achievable due to the inherent architectural trade-offs in high-throughput serving engines like SGLang. Benchmarks must therefore be designed to evaluate semantic correctness and statistical stability rather than asserting identical outputs across runs.

Based on these resolved architectural questions, the immediate scaffolding of the specified helper scripts and unit tests is recommended. The path forward is clear, and the findings herein provide the necessary technical clarity to proceed with a robust and reliable implementation.

## **Section 1: Strategic API Endpoint Analysis: /v1/chat/completions vs. /v1/responses in SGLang**

A foundational architectural decision for the benchmarking harness is the choice of the primary transport protocol for interacting with the SGLang server. The design document correctly identifies the two main OpenAI-compatible endpoints: /v1/chat/completions and /v1/responses. A thorough analysis of their respective maturity, feature completeness, and stability within the SGLang framework is essential for building a reliable tool.

### **1.1 Current State and Maturity of /v1/responses**

The SGLang codebase confirms the existence of an entrypoint for the OpenAI Responses API. The primary server module, http\_server.py, contains a v1\_responses\_request function and a corresponding ResponsesRequest Pydantic model for request validation.1 This indicates a deliberate effort to include this endpoint as part of SGLang's OpenAI compatibility layer.

However, its presence in the source code does not equate to functional maturity. Evidence from public issue trackers points to significant implementation gaps and instability. Users have reported fundamental Pydantic validation errors when using core parameters such as max\_new\_tokens, suggesting that the endpoint is not robustly tested or widely used.1 More critically, the endpoint fails to handle user-defined client-side tools, a key feature of the Responses API. One community discussion explicitly concludes that tool calling is likely supported only via the Chat Completions API, not the Responses API.1 These are not minor bugs but rather indicators of a feature that is, at best, a scaffold. For a benchmarking harness that prioritizes correctness and reliability, depending on such an experimental feature would introduce unacceptable risk.

### **1.2 Stability and Feature Completeness of /v1/chat/completions**

In stark contrast, the /v1/chat/completions endpoint represents the de facto industry standard for interacting with language models and is the most stable and well-supported API within SGLang. It is the primary interface demonstrated in official SGLang documentation, third-party deployment guides, and community examples for a wide range of models, including Qwen3.3

This endpoint reliably handles all core functionality required by the harness. Crucially, it supports the passing of non-standard, model-specific parameters through the extra\_body field in the request payload. This mechanism is the correct and documented way to control advanced features like Qwen's "thinking" mode, which will be detailed in Section 2\.7 While the Chat Completions API does not offer a native, structured field for returning model reasoning, it provides a stable and predictable foundation upon which client-side parsing logic can be built. For a tool that aims for broad compatibility and dependable operation, aligning with the most battle-tested and widely adopted endpoint is the most prudent engineering decision.

### **1.3 The "Compatibility Illusion" and Its Impact on Probing Strategy**

The existence of the /v1/responses endpoint in SGLang's source code creates a "compatibility illusion." While the API signature is present, the underlying implementation is not functionally complete. This is a common pattern in rapidly evolving open-source projects that aim to mirror the feature sets of proprietary APIs. This situation has a direct and critical impact on the proposed capability probing strategy.

The initial design document suggests a probe using HEAD, OPTIONS, or a trivial POST to determine if /v1/responses is supported. However, the evidence of its functional instability shows that such a probe is dangerously insufficient. The endpoint may well return a 200 OK status for a simple request while failing on any non-trivial payload that includes parameters like tool definitions or token limits.1 A simple probe would therefore produce a false positive, incorrectly flagging the feature as "supported" and causing the harness to switch to an unreliable transport.

To be effective, a capability probe must test for *functional correctness*, not just endpoint existence. This would require sending a more complex POST request that includes parameters known to be problematic and then rigorously validating both the HTTP status and the response schema. This dramatically increases the complexity and brittleness of the probe itself, turning it from a simple check into a mini-integration test that must be maintained. The broader implication is that building a stable benchmarking tool on bleeding-edge, non-core features of a serving engine is a high-risk strategy that undermines the goal of reliability.

### **1.4 Recommendation: Standardize on /v1/chat/completions**

Based on the overwhelming evidence of instability and functional incompleteness, the /v1/responses endpoint in SGLang must be considered experimental and is unfit for the stated goals of the observability harness.

The harness should be implemented to default to and exclusively use the /v1/chat/completions endpoint. The proposed logic for a capability probe should be removed from the design or deferred indefinitely. This decision will avoid introducing a complex and failure-prone dependency, ensuring the harness remains robust, maintainable, and aligned with the most stable features of the SGLang server. The following table summarizes the comparative analysis.

**Table 1: API Endpoint Feature Matrix**

| Feature | /v1/chat/completions | /v1/responses |
| :---- | :---- | :---- |
| **Stability** | ✅ Production Ready | ❌ Experimental, Known Bugs 1 |
| **Structured Reasoning** | ❌ Not Native (Requires client-side parsing) | ⚠️ Theoretically Supported (but unreliable) |
| **Tool Calling** | ✅ Supported | ❌ Known Issues with Custom Tools 1 |
| **Community Support** | ✅ High (Primary endpoint) | ❌ Low (Infrequent mentions, bug reports) |
| **Recommendation** | **Default and Only Transport** | **Avoid for Benchmarking** |

## **Section 2: Mastering "Thinking" Models: A Deep Dive into Qwen3-Next Configuration**

Effectively benchmarking "thinking" models like Qwen3-Next requires a precise understanding of the mechanisms used to control their reasoning behavior. This section details the specific server and request parameters necessary to manage Qwen's thinking mode within the SGLang environment, directly addressing the second open question.

### **2.1 Enabling and Disabling Thinking: The chat\_template\_kwargs Mechanism**

For hybrid-thinking models in the Qwen3 family, the reasoning mode is controlled at the request level via an enable\_thinking boolean parameter. This parameter is ultimately consumed by the tokenizer's apply\_chat\_template method on the server to format the prompt correctly, either including or omitting the directives that trigger the model's chain-of-thought process.9

When interacting with an OpenAI-compatible API server like SGLang, this model-specific parameter is not part of the standard API schema. Instead, it must be passed within a non-standard extra\_body object in the JSON request payload. Specifically, it is nested within a chat\_template\_kwargs dictionary.4 This confirms that the proposed \--qwen-enable-thinking command-line flag in infer\_client.py is the correct design, and its implementation should map directly to populating this structure.

A valid JSON request body to enable thinking for a Qwen3 model would be structured as follows:

JSON

{  
  "model": "Qwen/Qwen3-8B",  
  "messages": \[{"role": "user", "content": "Explain the principles of quantum entanglement."}\],  
  "extra\_body": {  
    "chat\_template\_kwargs": {  
      "enable\_thinking": true  
    }  
  }  
}

Setting enable\_thinking to false in this structure will disable the reasoning mode for that specific request, causing the model to generate a direct response without the \<think\>...\</think\> block.

### **2.2 Server-Side Requirement: The \--reasoning-parser Argument**

While the chat\_template\_kwargs parameter controls the model's *generation* of thinking content, a separate server-side configuration is required to instruct SGLang to *parse* this content into a structured format. To have SGLang automatically identify the \<think\>...\</think\> block and separate it from the final answer, the server must be launched with the \--reasoning-parser qwen3 command-line argument.4

When this flag is active, the JSON response from the /v1/chat/completions endpoint will be modified. The standard content field of the assistant's message will contain only the final, user-facing answer. The reasoning process will be placed in a separate, non-standard reasoning\_content field. If this flag is omitted, the server will return the entire raw output, including the \<think\>...\</think\> tags, within the single content field, leaving the parsing responsibility entirely to the client.

This server-side flag is therefore a crucial component for simplifying the client-side logic of the harness. The start\_server.sh script must be designed to conditionally append this argument when the model under test is known to be a Qwen thinking model.

### **2.3 Investigating reasoning\_effort and max\_reasoning\_tokens**

A comprehensive review of the SGLang documentation, Qwen model cards, and related community discussions reveals no evidence of API parameters named reasoning\_effort or a direct equivalent to max\_reasoning\_tokens.12 The control over the model's reasoning is binary (enable\_thinking) rather than granular.

The Qwen documentation suggests that the *depth* and *quality* of reasoning are properties of the model itself (e.g., specialized \-Thinking variants are designed for more complex tasks) and are influenced by providing a sufficient generation budget via the standard max\_tokens parameter.11 While some community members have explored client-side workarounds, such as using a logits processor to force a \</think\> token after a certain number of generated tokens, this is a custom client-side intervention, not a feature of the SGLang server API.8

The absence of a direct max\_reasoning\_tokens control knob is a critical finding. It signifies that the "thinking" process is not a distinct, pre-allocated stage of generation. Instead, it is an integral part of a single, continuous text generation process governed by the chat template. The tokens generated as part of the reasoning block are counted against the same max\_tokens limit as the final answer.

This introduces a key potential failure mode for the benchmark: **reasoning truncation**. If a test scenario sets the max\_tokens parameter too low (e.g., 256), the model might expend the entire token budget generating its reasoning and be forced to stop before producing any final content. The resulting API response would have a stop\_reason of length, and the content field might be empty or incomplete. Therefore, the benchmark harness must treat max\_tokens as a combined budget for both reasoning and content. Test configurations must use a sufficiently high max\_tokens value to prevent this failure mode, and the analysis of test results must carefully inspect cases where the stop reason is length to identify potential reasoning truncation.

## **Section 3: Implementing State-Preserving Multi-Turn Conversations**

A core requirement of the benchmark harness is to correctly handle multi-turn conversations, particularly with models that exhibit complex behaviors like chain-of-thought reasoning. The strategy for managing conversational history is critical for ensuring that the model receives the correct context in subsequent turns.

### **3.1 Official Guidance: Exclude Reasoning from History**

The proposed continuation rule in the design document—to exclude prior assistant reasoning from subsequent turns—is definitively validated by official model provider documentation. The Qwen documentation is unequivocal on this point: "In multi-turn conversations, the historical model output should only include the final output part and does not need to include the thinking content".15 This guidance is reiterated in community-authored best practice guides.17

This principle is not unique to Qwen. Documentation for other reasoning models, such as DeepSeek-R1, provides similar warnings, explicitly stating that the reasoning\_content field from an API response should not be fed back to the model in subsequent prompts.18 Including the reasoning block in the conversation history can confuse the model, leading to bloated, off-topic, or repetitive responses in later turns.

### **3.2 Correct messages Array Construction**

The correct implementation of this continuation rule requires a clear separation between the complete conversation transcript stored by the client and the payload sent to the API. The client-side transcript.json artifact should serve as the immutable, comprehensive record of the entire interaction, storing the system prompt, all user turns, and both the assistant\_reasoning and assistant\_content for each assistant turn.

However, when constructing the messages array for the *next* API call, the build\_messages\_from\_transcript helper function must filter this history. It should iterate through the transcript and append messages in chronological order, but for assistant turns, it must only include the content field.

The flow for a two-turn conversation should proceed as follows:

1. **Turn 1 Request:** The messages array contains only the initial user prompt.  
   * messages \=  
2. **Turn 1 Response:** The server returns both reasoning and content (assuming \--reasoning-parser is enabled).  
   * reasoning\_content: "\<think\>The user is asking a factual question...\</think\>", content: "The capital of France is Paris."  
3. **Client-side Transcript Update:** The client stores the full, structured response in its transcript.json log.  
4. **Turn 2 Request:** The build\_messages\_from\_transcript function constructs a new messages array. It includes the first user turn, the *content only* from the first assistant turn, and the new user turn.  
   * messages \=

This pattern ensures the model receives a clean, coherent conversational history that adheres to the format it was trained on, providing a solid foundation for your build\_messages\_from\_transcript helper function.

### **3.3 Transcript as the Source of Truth**

The necessity of filtering the conversation history before sending it to the API highlights a crucial architectural pattern: the separation of the "complete conversation log" from the "API request payload." A naive implementation might simply append the raw response object from the server to a list and attempt to resend it, but as the research confirms, this approach is incorrect for thinking models.15

The application must therefore maintain two distinct representations of the conversation. The transcript.json artifact is the immutable, observable record of everything that occurred during the test, including all metadata and the model's internal reasoning. It is the canonical source of truth for post-run analysis and debugging. In contrast, the messages array is a transient, filtered representation specifically crafted for the model's consumption in the next turn of the conversation.

This reinforces the importance of the transcript.json artifact as designed. It is not merely a log file; it is the persistent state of the conversation from which all future API request payloads are derived. The harness tooling should be built with this clear distinction in mind.

## **Section 4: Comprehensive Observability: Token Accounting and Performance Metrics**

A primary goal of the harness is to capture detailed and accurate metrics. This requires a clear understanding of what data the SGLang server provides reliably and what must be calculated or inferred by the client, particularly for token usage and latency.

### **4.1 Token Accounting Across Endpoints**

The reliability of token counts varies depending on the API endpoint and the server configuration.

For the recommended /v1/chat/completions endpoint, the standard usage object in the response reliably provides prompt\_tokens and completion\_tokens. The prompt\_tokens value corresponds to the tokenized input messages array. The completion\_tokens value represents the total number of tokens generated by the model in its response. Crucially, when thinking is enabled, this completion\_tokens count is a sum; it includes the tokens for both the reasoning block and the final content. Even when the \--reasoning-parser flag is used, SGLang does not add a separate reasoning\_tokens field to the usage object.

While the unstable /v1/responses API is designed to potentially surface a num\_reasoning\_tokens field, its general unreliability makes this a purely theoretical advantage that cannot be depended upon for the harness.

This leads to a significant implementation detail for the client-side harness. To accurately measure and log reasoning\_tokens and content\_tokens as separate metrics when using the stable /v1/chat/completions endpoint, the benchmarking client *must* have access to the model's specific tokenizer. The server provides the reasoning\_content and content as raw strings, but not their token counts. The only way to obtain these counts is for the client to tokenize these strings itself.

This introduces a new dependency for the infer\_runner.py script. It must be capable of loading the correct Hugging Face tokenizer corresponding to the MODEL\_PATH being used by the SGLang server. A robust implementation would involve initializing a tokenizer via transformers.AutoTokenizer.from\_pretrained(model\_path) at the start of a test run. This allows for the client-side calculation of the disaggregated token counts, but it also introduces a potential point of failure if the client environment cannot access the tokenizer or if there is a version mismatch. The harness design must account for this dependency.

### **4.2 Streaming Semantics and Latency Measurement with Thinking**

SGLang supports streaming generation, which is a planned future feature for the harness.12 When streaming a response from a thinking model, the tokens corresponding to the \<think\>...\</think\> block will be part of the streamed delta chunks, arriving before the tokens for the final answer.

This behavior makes the standard "Time to First Token" (TTFT) metric ambiguous and potentially misleading. A naive TTFT measurement would capture the time until the very first token (likely the \< of \<think\>) is received. While this is a valid measure of the model's initial response time, it does not represent the user-perceived latency for receiving the actual answer.

To provide a more nuanced and useful performance picture, the benchmark should aim to capture two distinct first-token latency metrics when streaming is enabled:

* **first\_reasoning\_token\_latency\_ms**: The time elapsed from sending the request until the first token of the reasoning block is received. This metric quantifies the model's "time to begin thinking."  
* **first\_content\_token\_latency\_ms**: The time elapsed from sending the request until the first token *after* the closing \</think\> tag is received. This metric quantifies the user-perceived "time to answer."

Implementing this requires the streaming client to parse the incoming token chunks in real-time to identify the \</think\> boundary. This adds complexity to the client but yields far more insightful performance data.

The following table provides a clear guide for the metrics collection logic, detailing what can be sourced from the server versus what must be calculated by the client.

**Table 2: Token Accounting Reliability by Endpoint**

| Metric | /v1/chat/completions Source | /v1/responses Source | Reliability |
| :---- | :---- | :---- | :---- |
| prompt\_tokens | ✅ Server (usage object) | ✅ Server (presumed) | High |
| completion\_tokens | ✅ Server (usage object) | ✅ Server (presumed) | High (Note: Includes reasoning tokens) |
| reasoning\_tokens | ⚠️ **Client-side calculation required** | ❓ Potentially Server (but API is unstable) | Low / Requires extra dependency |
| content\_tokens | ⚠️ **Client-side calculation required** | ❓ Potentially Server (but API is unstable) | Low / Requires extra dependency |

## **Section 5: Achieving Reproducibility: A Realistic Assessment of Determinism in SGLang**

Reproducibility is a desirable characteristic for any benchmarking tool, as it allows for consistent testing and regression analysis. The design document correctly identifies the seed parameter as a potential mechanism for achieving deterministic outputs. However, a deep analysis of SGLang's architecture reveals fundamental limitations to this approach.

### **5.1 The seed Parameter in SGLang**

The SGLang API and its associated benchmarking scripts do accept a seed parameter in the request payload.19 This aligns with the OpenAI API specification, where the seed parameter is intended to make outputs "mostly" deterministic.20 However, practical experience with SGLang, as documented in a public GitHub issue, confirms that using a fixed seed still results in varying outputs across identical requests.22

This behavior is not a bug but an expected consequence of the system's design. The official SGLang FAQ provides a clear explanation: "The results are not deterministic, even with a temperature of 0".23 The document attributes this non-determinism to two core architectural features of high-performance inference servers:

1. **Dynamic Batching:** To maximize GPU utilization, SGLang groups incoming requests into dynamic batches for processing. The size and composition of these batches can vary depending on request arrival times and sequence lengths. Different batch compositions can cause PyTorch and its underlying CUDA libraries to dispatch different computational kernels, which may have slight numerical differences in their floating-point operations.  
2. **Prefix Caching (RadixAttention):** SGLang's RadixAttention mechanism reuses the KV cache for shared prefixes across different requests.12 The state of this cache and how it is accessed can introduce subtle variations.

These small numerical differences, while insignificant in isolation, can cascade and amplify as they propagate through the many layers of a large language model, ultimately resulting in a different sequence of generated tokens.

### **5.2 Redefining "Correctness" for Benchmarking**

The architectural trade-off made by SGLang—prioritizing throughput and efficiency over strict determinism—fundamentally alters the definition of a "correctness-first" benchmark. If correctness is defined as bit-for-bit, identical output on every run, then any benchmark built on SGLang is destined to fail. This premise is flawed.

Therefore, the concept of "correctness" for this harness must be redefined to be more robust and realistic. Instead of simple string comparison, correctness evaluation should focus on higher-level attributes:

* **Semantic Correctness:** Does the model's response accurately and appropriately address the intent of the prompt? This may require programmatic checks or, for more complex cases, evaluation by a more powerful "judge" LLM.  
* **Format Adherence:** If the prompt requested a specific format (e.g., valid JSON, a numbered list), does the output conform to that format? This can be validated programmatically.  
* **Constraint Following:** Does the model successfully adhere to any negative constraints or other specific instructions provided in the prompt?

This shift has significant implications for the design of the harness. The infer\_summarize.py tool becomes more than just a metrics aggregator; it must evolve into an evaluation engine. For quantitative metrics like latency and token counts, the goal should be to measure statistical stability and analyze distributions across multiple runs, rather than expecting identical values.

### **5.3 Impact on Reasoning**

The inherent non-determinism of the generation process naturally extends to the model's reasoning. The exact phrasing, structure, and path of the model's "thought" process within the \<think\>...\</think\> block can vary between runs, even if it ultimately arrives at the same final conclusion.

When analyzing reasoning artifacts, the focus should be on the substantive logical steps and the overall quality of the reasoning, rather than on the exact token sequence. The seed parameter should still be used in all test requests, as it serves to *reduce* variability, making the outputs more consistent than they would be otherwise. However, it must be treated as a tool to encourage, not guarantee, reproducibility.

## **Section 6: Advanced Implementation Guidance and Recommendations**

This final section synthesizes the preceding analysis into a set of concrete, actionable recommendations to guide the implementation of the SGLang observability harness.

### **6.1 Refined Server Startup Command**

The start\_server.sh script should be constructed to be flexible and incorporate best practices for performance and memory management. It should accept variables for key parameters and conditionally add flags based on the model being tested. A robust template for the core sglang.launch\_server command is as follows:

Bash

\# Example invocation within start\_server.sh

\# Set a default, but allow override  
REASONING\_PARSER\_FLAG=""  
if\]; then  
  REASONING\_PARSER\_FLAG="--reasoning-parser qwen3"  
fi

python \-m sglang.launch\_server \\  
  \--model-path "$MODEL\_PATH" \\  
  \--port "$PORT" \\  
  \--tp-size "$TP" \\  
  \--mem-fraction-static "$MEM\_FRACTION\_STATIC" \\  
  \--context-length "$CONTEXT\_LENGTH" \\  
  \--chunked-prefill-size "$CHUNKED\_PREFILL\_SIZE" \\  
  \--trust-remote-code "$TRUST\_REMOTE\_CODE" \\  
  ${REASONING\_PARSER\_FLAG}

Special attention should be paid to memory-related parameters, especially when conducting long-context tests. Parameters like \--mem-fraction-static (which controls the size of the KV cache pool) and \--chunked-prefill-size (which breaks up long prompts to avoid OOM errors during prefill) are critical levers for tuning server stability.23

### **6.2 Detailed Schema for Artifacts**

To ensure consistent and machine-readable outputs, the JSON artifacts generated by infer\_runner.py should adhere to a well-defined schema. Based on the findings of this report, the following schema is recommended for the metrics.json file, clearly delineating between server-provided and client-calculated values.

**Recommended metrics.json Schema:**

JSON

{  
  "usage": {  
    "prompt\_tokens": 125,               // From server's 'usage' object  
    "completion\_tokens": 512,           // From server's 'usage' object (total generated)  
    "reasoning\_tokens": 350,            // Calculated by client via tokenizer  
    "content\_tokens": 162,              // Calculated by client via tokenizer  
    "total\_tokens": 637                 // Sum of prompt\_tokens and completion\_tokens  
  },  
  "timing": {  
    "total\_latency\_ms": 3450.7,         // End-to-end client-side measurement  
    "first\_reasoning\_token\_latency\_ms": 450.2, // If streaming is enabled  
    "first\_content\_token\_latency\_ms": 2100.5   // If streaming is enabled  
  },  
  "request": {  
    "model": "Qwen/Qwen3-8B",  
    "temperature": 0.2,  
    "max\_tokens": 1024,  
    "seed": 42,  
    "extra\_body": {  
      "chat\_template\_kwargs": {  
        "enable\_thinking": true  
      }  
    }  
  },  
  "response": {  
    "id": "chatcmpl-...",  
    "stop\_reason": "stop",  
    "system\_fingerprint": "fp\_..."  
  },  
  "server\_metadata": {  
    "model\_path": "/path/to/Qwen/Qwen3-8B",  
    "tp\_size": 1,  
    "mem\_fraction\_static": 0.9  
  },  
  "run\_metadata": {  
    "container\_run\_id": "run-20240926-103000-xyz",  
    "test\_id": "mt1"  
  }  
}

### **6.3 Final Recommendation on Scaffolding**

The research and analysis conducted for this report have successfully resolved the most critical architectural and implementation questions facing the project. The choice of /v1/chat/completions as the sole transport layer is clear and well-supported. The mechanisms for controlling model "thinking," the correct procedure for conversation continuation, and the realistic limitations of determinism are now understood and can be accounted for in the design.

There are no remaining open questions that would necessitate a significant architectural pivot. Therefore, it is strongly recommended to **proceed with the scaffolding of the helper scripts, client tools, and unit tests immediately**. The path forward is well-defined, and the principles outlined in this report provide a solid foundation for building a robust, insightful, and reliable SGLang observability harness.

#### **Works cited**

1. Clarify support for tool calling and OpenAI Responses API · Issue \#10038 · sgl-project/sglang \- GitHub, accessed October 11, 2025, [https://github.com/sgl-project/sglang/issues/10038](https://github.com/sgl-project/sglang/issues/10038)  
2. \[Bug\] \`/v1/responses\` raises an error when max\_new\_tokens is not None and doesn't work with non-builtin tools. · Issue \#10014 · sgl-project/sglang \- GitHub, accessed October 11, 2025, [https://github.com/sgl-project/sglang/issues/10014](https://github.com/sgl-project/sglang/issues/10014)  
3. runpod-workers/worker-sglang: SGLang is fast serving framework for large language models and vision language models. \- GitHub, accessed October 11, 2025, [https://github.com/runpod-workers/worker-sglang](https://github.com/runpod-workers/worker-sglang)  
4. SGLang \- Qwen, accessed October 11, 2025, [https://qwen.readthedocs.io/en/latest/deployment/sglang.html](https://qwen.readthedocs.io/en/latest/deployment/sglang.html)  
5. SGLang Quick Start Guide — Gaudi Documentation 1.22.1 documentation, accessed October 11, 2025, [https://docs.habana.ai/en/latest/PyTorch/SGLang\_Inference/SGLang\_Quick\_Start.html](https://docs.habana.ai/en/latest/PyTorch/SGLang_Inference/SGLang_Quick_Start.html)  
6. In-Depth: Deploy with SGLang \- Docs \- DataCrunch, accessed October 11, 2025, [https://docs.datacrunch.io/containers/tutorials/deploy-with-sglang-indepth](https://docs.datacrunch.io/containers/tutorials/deploy-with-sglang-indepth)  
7. Deployment example for a qwen3 model with hybrid thinking \- vLLM Forums, accessed October 11, 2025, [https://discuss.vllm.ai/t/deployment-example-for-a-qwen3-model-with-hybrid-thinking/1462](https://discuss.vllm.ai/t/deployment-example-for-a-qwen3-model-with-hybrid-thinking/1462)  
8. In Qwen 3 you can use /no\_think in your prompt to skip the reasoning step \- Reddit, accessed October 11, 2025, [https://www.reddit.com/r/LocalLLaMA/comments/1kasjge/in\_qwen\_3\_you\_can\_use\_no\_think\_in\_your\_prompt\_to/](https://www.reddit.com/r/LocalLLaMA/comments/1kasjge/in_qwen_3_you_can_use_no_think_in_your_prompt_to/)  
9. Qwen/Qwen3-32B \- Hugging Face, accessed October 11, 2025, [https://huggingface.co/Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)  
10. Qwen/Qwen3-8B \- Hugging Face, accessed October 11, 2025, [https://huggingface.co/Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)  
11. Quickstart \- Qwen \- Read the Docs, accessed October 11, 2025, [https://qwen.readthedocs.io/en/latest/getting\_started/quickstart.html](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html)  
12. haotian-liu/sglang\_contrib: SGLang is a structured generation language designed for large language models (LLMs). It makes your interaction with LLMs faster and more controllable. \- GitHub, accessed October 11, 2025, [https://github.com/haotian-liu/sglang\_contrib](https://github.com/haotian-liu/sglang_contrib)  
13. sgl-project/sglang: SGLang is a fast serving framework for large language models and vision language models. \- GitHub, accessed October 11, 2025, [https://github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)  
14. Qwen3 is the large language model series developed by Qwen team, Alibaba Cloud. \- GitHub, accessed October 11, 2025, [https://github.com/QwenLM/Qwen3](https://github.com/QwenLM/Qwen3)  
15. Qwen/Qwen3-4B-Thinking-2507 \- Hugging Face, accessed October 11, 2025, [https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)  
16. Limiting Qwen 3's Thinking \- Zach Mueller, accessed October 11, 2025, [https://muellerzr.github.io/til/end\_thinking.html](https://muellerzr.github.io/til/end_thinking.html)  
17. Best Settings to Run Qwen3-30B-A3B Locally \- Jan.ai, accessed October 11, 2025, [https://www.jan.ai/post/qwen3-settings](https://www.jan.ai/post/qwen3-settings)  
18. DeepSeek Prompting Techniques: strategies, limits, best practices, etc \- Data Studios, accessed October 11, 2025, [https://www.datastudios.org/post/deepseek-prompting-techniques-strategies-limits-best-practices-etc](https://www.datastudios.org/post/deepseek-prompting-techniques-strategies-limits-best-practices-etc)  
19. sglang/python/sglang/bench\_offline\_throughput.py · tuandunghcmut/vlm\_clone\_2 at 8c911bfbe92c48ef2a5ca349a2b17473f6840280 \- Hugging Face, accessed October 11, 2025, [https://huggingface.co/tuandunghcmut/vlm\_clone\_2/blob/8c911bfbe92c48ef2a5ca349a2b17473f6840280/sglang/python/sglang/bench\_offline\_throughput.py](https://huggingface.co/tuandunghcmut/vlm_clone_2/blob/8c911bfbe92c48ef2a5ca349a2b17473f6840280/sglang/python/sglang/bench_offline_throughput.py)  
20. How to Use the Seed parameter? \- Vellum AI, accessed October 11, 2025, [https://www.vellum.ai/llm-parameters/seed](https://www.vellum.ai/llm-parameters/seed)  
21. How to make your completions outputs consistent with the new seed parameter, accessed October 11, 2025, [https://cookbook.openai.com/examples/reproducible\_outputs\_with\_the\_seed\_parameter](https://cookbook.openai.com/examples/reproducible_outputs_with_the_seed_parameter)  
22. \[Bug\] Non-deterministic outputs with fixed seed parameter in API server · Issue \#5377 · sgl-project/sglang \- GitHub, accessed October 11, 2025, [https://github.com/sgl-project/sglang/issues/5377](https://github.com/sgl-project/sglang/issues/5377)  
23. Troubleshooting and Frequently Asked Questions — SGLang, accessed October 11, 2025, [https://sgl-project.github.io/references/faq.html](https://sgl-project.github.io/references/faq.html)  
24. Inference Using SGLang — Gaudi Documentation 1.22.1 documentation, accessed October 11, 2025, [https://docs.habana.ai/en/latest/PyTorch/SGLang\_Inference/SGLang\_Inference.html](https://docs.habana.ai/en/latest/PyTorch/SGLang_Inference/SGLang_Inference.html)  
25. sglang 0.3.1.post2 \- PyPI, accessed October 11, 2025, [https://pypi.org/project/sglang/0.3.1.post2/](https://pypi.org/project/sglang/0.3.1.post2/)