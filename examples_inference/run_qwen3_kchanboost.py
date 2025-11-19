# examples/run_qwen3_kchanboost.py

from typing import Optional
import time
import shutil
import gc
import torch
from transformers import AutoTokenizer, PreTrainedModel
from transformers.models.qwen3 import Qwen3Config
from transformers.models.qwen3 import Qwen3ForCausalLM
from kchanboost.models.qwen3 import Qwen3ForCausalLM_KChanBoost
from kchanboost.kvcache import KChanBoostCache, get_kvcache_kchanboost
#

def get_prompt(prompt_choice: int) -> tuple[str, str]:
    if prompt_choice == 1:
        task_name = "gsm8k"
        prompt = """"Given the following problem, think step by step and give a final answer to the problem.
        Problem: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
        Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
        There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6
        Given the following problem, think step by step and give a final answer to the problem.
        Problem: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
        Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
        There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5
        Given the following problem, think step by step and give a final answer to the problem.
        Problem: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
        Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
        Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39
        Given the following problem, think step by step and give a final answer to the problem.
        Problem: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
        Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
        Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8
        Given the following problem, think step by step and give a final answer to the problem.
        Problem: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
        Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
        Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9
        Given the following problem, think step by step and give a final answer to the problem.
        Problem: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
        Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
        There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29
        Given the following problem, think step by step and give a final answer to the problem.
        Problem: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
        Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
        Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33
        Given the following problem, think step by step and give a final answer to the problem.
        Problem: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
        Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
        Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8

        Given the following problem, think step by step and give a final answer to the problem.
        Problem: Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
        Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
        """
    elif prompt_choice == 2:
        task_name = "gpqa"
        prompt = """"Here are some example questions from experts. Think step by step and then give the final answer, following the format of the previous questions exactly.
        Question: In a given population, 1 out of every 400 people has a cancer caused by a completely recessive allele, b. Assuming the population is in Hardy-Weinberg equilibrium, which of the following is the expected proportion of individuals who carry the b allele but are not expected to develop the cancer?
        Choices:
        (A) 1/400
        (B) 19/400
        (C) 20/400
        (D) 38/400
        Let's think step by step:  The expected proportion of individuals who carry the b allele but are not expected to develop the cancer equals to the frequency of heterozygous allele in the given population. According to the Hardy-Weinberg equation p^2 + 2pq + q^2 = 1, where p is the frequency of dominant allele frequency, q is the frequency of recessive allele frequency, p^2 is the frequency of the homozygous dominant allele, q^2 is the frequency of the recessive allele, and 2pq is the frequency of the heterozygous allele. Given that q^2=1/400, hence, q=0.05 and p=1-q=0.95. The frequency of the heterozygous allele is 2pq=2*0.05*0.95=38/400. The final answer is (D).

        Question: A Fe pellet of 0.056 g is first dissolved in 10 mL of hydrobromic acid HBr (0.1 M). The resulting solution is then titrated by KMnO4 (0.02 M). How many equivalence points are there?
        Choices:
        (A) Two points, 25 ml and 35 ml
        (B) One point, 25 mL
        (C) One point, 10 ml
        (D) Two points, 25 ml and 30 ml
        Let's think step by step:  HBr will react with Fe to produce Fe2+. MnO4- will first react with Fe2+ then Br-. Two equivalence points will exist 25 ml and 35 ml. In the beaker there is Fe2+ and Br-. When considering titration with two analytes one will have to consider which reaction will occur first. Since it is a redox titration consider the reduction potential of: E0 (Br2 /Br- ) = 1.09 V  \tE0 (MnO4-/ Mn2+) = 1.49 V\tE0 (Fe3+/Fe2+) =0.77 V. [Fe2+]=m/MV=0.1M. Reaction 1: MnO4- + 5Fe2+ + 8H+ \u2192 Mn2+ + 5Fe3+ + 4H2O. Reaction 2: 2MnO4- + 10Br- + 16H+ \u2192 2Mn2+ + 5Br2 + 8H2O. So MnO4- will first react with Fe2+ with a stoichiometry of 1:5 so Veq1 will be 10 ml. Then when Fe2+ is used up, MnO4- will react with Br- with a stoichiometry of 2:10 then V added will be 25 ml so Veq2=25+10=35 ml. The final answer is (A).

        Question: Consider a quantum mechanical system containing a particle of mass $m$ moving in an isotropic three-dimensional potential of the form $V(r) = 1/2 m \u03c9^2 r^2$ corresponding to the acted force obeying Hooke's law. Here, $\u03c9$ is the angular frequency of oscillation and $r$ is the radial distance of the particle from the origin in spherical polar coordinates. What is the value of energy of the third excited state, and how many linearly independent eigenfunctions are possible for the same energy eigenvalue?
        Choices:
        (A) 11 \u03c0\u00b2 \u210f\u00b2 / (2m r\u00b2), 3
        (B) (9/2) \u210f \u03c9, 10
        (C) 11 \u03c0\u00b2 \u210f\u00b2 / (2m r\u00b2), 10
        (D) (9/2) \u210f \u03c9, 3
        Let's think step by step:  This problem is nothing but the three-dimensional simple harmonic oscillator (SHO) problem. The energy spectrum of three-dimensional SHO is $E_n= (n+3/2)\u210f \u03c9$ where $n=0,1,2,3\u2026$. For third excited state n=3. 3+3/2=6/2+3/2=9/2. Thus the corresponding energy is $(9/2)\u210f \u03c9$. The degeneracy of the state is $g_n= (n+1)(n+2)/2$. For n=3, degeneracy is (3+1)*(3+2)/2=4*5/2=10. The final answer is (B).

        Question: Your overhear two chemists talking to each other as they leave a synthetic organic chemistry lab. One asks the other 'So, how did it go?' The second chemist replies, 'Not well - my compounds are on top of each other.' What is the second chemist most likely referring to?
        Choices:
        (A) The compounds they are working with have similar polarities.
        (B) The compounds they are working with have similar boiling points.
        (C) The compounds they are working with are bonding to each other through non-covalent/van der Waals interactions.
        (D) The compounds they are working with have similar optical rotations.
        Let's think step by step:  On top of each other commonly refers to two compounds that have similar Rf values on chromatography (a common operation in synthetic chemistry). Similar Rf values arise for compounds with similar polarities. The final answer is (A).

        Question: Two people are playing the following game. A fair coin is tossed into the air. Person A says that in a single toss of the coin, the tail will come. So it's like the first shot or the third shot or the fifth shot. Person B says that the coin will come with a double toss. So like the second, fourth, sixth or eighth shot. Imagine this game played forever. What is the probability that person A wins this game?
        Choices:
        (A) 1/2
        (B) 1/4
        (C) 2/3
        (D) 1/8
        Let's think step by step:  When finding the correct answer, the probability of playing forever and the coin's single-point toss will be calculated. For example, a tail may appear on the first shot. This probability is 1/2. if the first toss doesn't come up, it shouldn't come to the second roll either, because the second throw is an even number. So it can come in the third shot. This is (1/2)(1/2)(1/2). So (1/2)^3=1/8. Or it could come on the fifth shot. This is (1/2)^5=1/32. This is actually a geometric series that goes on forever. We can write this series as follows. (1/2) + (1/2)^3 + (1/2)^5 + (1/2)^7 + \u2026\u2026\u2026. The solution for this series is as follows : a1/(1-r) where a1 is the first number and r is the sequence or r= a2/a1 or a3/a2 etc. a1=1/2 r=(1/2)^2=1/4 So a1/(1-r)=(1/2)/(1-1/4)=(1/2)/(3/4)=2/3. The final answer is (C).

        Question: Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?
        Choices:
        (A) 10^-9 eV
        (B) 10^-11 eV
        (C) 10^-8 eV
        (D) 10^-4 eV
        Let's think step by step: ",
        """
    else:
        task_name = "general"
        prompt = "现在父亲的年龄是儿子的 3 倍。再过 15 年后，父亲的年龄会是儿子的 2 倍。请回答以下两个问题： 1.现在父亲几岁？ 2.现在儿子几岁？"
    return task_name, prompt


#
import argparse
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Qwen/Qwen3-8B", help='Qwen3 model to load')
    parser.add_argument("--fp16_kv_cache",   action='store_true',      help="Use the default KV cache implementation from Transformers")
    parser.add_argument("--max_token_new",   type=int, default=200,    help="Maximum number of new tokens to generate")
    parser.add_argument("--max_seq_len",     type=int, default=8192,   help="Maximum sequence length for KV cache, should be larger than the prompt length + max_token_new")
    parser.add_argument("--batch_size",      type=int, default=1,      help="Batch size for generation, repeat the prompt for each batch")
    parser.add_argument("--prompt_choice",   type=int, default=0,      help="Choice of prompt to use")
    return parser


def model_generate(model: PreTrainedModel, tokenizer: AutoTokenizer, inputs: dict, max_new_tokens=200, kv_cache: Optional[KChanBoostCache] = None):
    # Time consumption
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    # Execution
    outputs = model.generate(
        input_ids=inputs.input_ids.cuda(),
        attention_mask=inputs.attention_mask.cuda() if "attention_mask" in inputs else None,
        max_new_tokens=max_new_tokens, # max_new_tokens is the maximum number of new tokens to generate.
        return_dict_in_generate=True,
        past_key_values=kv_cache,
        do_sample=False,
        max_length=None,  # No maximum length, let the model decide based on max_new_tokens
        temperature=None,  # No temperature for deterministic generation
        top_p=None,       # No top-p sampling
        top_k=None,       # No top-k sampling
    )
    # Time consumption
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Model.generate() execution time: {elapsed_time:.4f} seconds.")
    # Decode the output sequences
    output_sequences = outputs["sequences"]
    output_texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    # Print outputs
    width = shutil.get_terminal_size().columns
    print('-' * width)
    for i, text in enumerate(output_texts):
        print(f"\n======== BATCH {i} OUTPUT ========\n")
        print(text)
    print('-' * width)
    #
    return outputs


def main() -> None:
    args = build_parser().parse_args()
    print("Model: ",args.model)
    #
    prompt_choice = args.prompt_choice
    task_name, prompt = get_prompt(prompt_choice)
    print(f"Task: {task_name}, Prompt: {prompt}")
    #
    prompt = [prompt for _ in range(args.batch_size)]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    #
    APPLY_CHAT_TEMPLATE = True
    if APPLY_CHAT_TEMPLATE:
        messages = [[{"role": "user", "content": p}] for p in prompt]
        texts = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        inputs = tokenizer(text=texts, return_tensors="pt")
    else:
        inputs = tokenizer(text=prompt, return_tensors="pt")
    #
    config = Qwen3Config.from_pretrained(args.model)
    #
    if args.fp16_kv_cache:
        print("Using the default KV cache implementation from Transformers.")
        kv_cache = None
        model_class = Qwen3ForCausalLM
    else:
        print("Using KChanBoost KV cache implementation.")
        kv_cache = get_kvcache_kchanboost(
            config,
            args.batch_size,
            args.max_seq_len
        )
        model_class = Qwen3ForCausalLM_KChanBoost
    model = model_class.from_pretrained(
        pretrained_model_name_or_path=args.model,
        config = config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        device_map='auto'
    )
    model.eval()
    #
    outputs = model_generate(model, tokenizer, inputs, args.max_token_new, kv_cache)
    #
    model.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()