"""One realistic prompt per task. Generation cap is 6144 tokens."""

TASKS = [
    (
        "coding",
        "Write a Python module that implements a streaming JSON parser. "
        "Include the parser, unit tests for truncated input, nested objects, "
        "strings with escapes, and a short README section on how to use it. "
        "Keep going until the tests and the usage notes are complete.",
    ),
    (
        "math",
        "Solve this step by step and do not skip algebra. "
        "A tank contains 1200 L of water. Brine with 0.2 kg/L salt enters at "
        "8 L/min and the well-mixed solution leaves at 8 L/min. Initially the "
        "tank has 15 kg of salt. Derive the ODE, solve it, find the salt "
        "after 30 minutes, and then generalize to inflow rate a, outflow rate "
        "b, concentration c, and initial salt s0. Explain each step in full.",
    ),
    (
        "qa",
        "Explain how attention, the feed-forward network, and residual "
        "connections interact in a transformer decoder layer. Then compare "
        "dense FFN with a top-8 mixture-of-experts FFN: what is computed, "
        "what is saved, and where the speed and quality tradeoffs come from. "
        "Write a thorough answer a new graduate student could study from.",
    ),
    (
        "summarization",
        "Summarize the following notes into a structured briefing with "
        "background, method, results, and open questions.\n\n"
        "The lab trained a 5-layer dense draft model on hidden states taken "
        "from layers 1, 13, 25, 37 and the final state of a 48-layer "
        "mixture-of-experts target. The draft has hidden size 2048, 32 heads, "
        "4 kv heads, and intermediate size 6144. Training used 1.42 million "
        "conversations, sequence length 3072, 8 nodes and 64 NPUs, AdamW, "
        "learning rate 6e-4, 5 epochs. Validation loss ended at 0.554 and "
        "full-token accuracy at 0.414. A later serving test at tensor "
        "parallel 4 and 48 concurrent requests showed prompt throughput near "
        "4100 tokens/s. With speculative decoding the mean acceptance length "
        "was about 2.1 out of 15 drafted tokens, so end-to-end time did not "
        "drop. The notes also mention that host-side writes stalled the cube "
        "during hidden-state extraction, and that checkpoint files were "
        "written only by the rank-0 process on 10.1.0.131.",
    ),
    (
        "translation",
        "Translate the following into clear, formal Chinese. Preserve the "
        "technical terms and the paragraph breaks.\n\n"
        "Speculative decoding runs a small draft model ahead of a large "
        "target model. The draft proposes several future tokens in one step. "
        "The target then verifies those tokens in parallel and keeps the "
        "longest prefix that matches its own distribution. If the draft is "
        "accurate, the target accepts many tokens per forward pass and the "
        "generation throughput rises. If the draft is inaccurate, most "
        "proposals are rejected, the extra draft compute is wasted, and the "
        "system can be slower than ordinary decoding. Acceptance length is "
        "the average number of tokens committed per verification, including "
        "the one token the target always produces.",
    ),
    (
        "writing",
        "Write a long technical blog post, about the length of a full essay, "
        "on how a research group would train and serve a speculative-decoding "
        "draft for a mixture-of-experts assistant. Cover data generation, "
        "which hidden states to store, what the draft architecture should "
        "copy from the target, how to read acceptance-length curves, and "
        "when speculative decoding is not worth turning on. Use concrete "
        "numbers where they help.",
    ),
    (
        "tool_call",
        "You can call tools by writing one JSON object per line with keys "
        "name and arguments. Available tools: search_papers(query, year_from), "
        "lookup_constant(name), run_python(code), unit_convert(value, from, to). "
        "The user asks: estimate the escape velocity of Mars, compare it with "
        "Earth, and cite two papers on Mars sample return after 2020. "
        "Plan the tool calls, write the calls, then write the final answer "
        "you would give after the tools returned plausible results.",
    ),
    (
        "science",
        "Explain, at the level of a review article section, how CRISPR-Cas9 "
        "finds a target, cuts DNA, and how base editing and prime editing "
        "differ from a double-strand break. Include the role of the PAM, "
        "the guide RNA, and one concrete experimental readout a lab would "
        "use to measure off-target edits. Write the full explanation.",
    ),
]
