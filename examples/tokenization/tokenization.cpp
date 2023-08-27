#include "common.h"
#include "llama.h"
#include "build-info.h"

#include <ctime>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

int main(int argc, char ** argv) {
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    params.embedding = false;

    if (params.n_ctx > 16384) {
        fprintf(stderr, "%s: warning: model does not support context sizes greater than 16384 tokens (%d specified);"
                "expect poor results\n", __func__, params.n_ctx);
    }

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    llama_backend_init(params.numa);

    llama_model * model;
    llama_context * ctx;

    // load the model
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    // tokenize the prompt
    auto embd_inp = ::llama_tokenize(ctx, params.prompt, true);

    if (params.verbose_prompt) {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        fprintf(stderr, "\n");
    }

    for (int i = 0; i < (int) embd_inp.size(); i++) {
      printf("%6d %s\n", embd_inp[i], llama_token_to_str(ctx, embd_inp[i]));
    }
    printf("\n");

    llama_print_timings(ctx);
    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
