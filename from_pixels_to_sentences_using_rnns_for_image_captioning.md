# From Pixels to Sentences: Using RNNs for Image Captioning

## Why RNNs Still Matter for Vision Tasks

Recurrent Neural Networks (RNNs) maintain a hidden state that evolves over time, enabling them to model sequences [Source](https://www.youtube.com/watch?v=kG2lAPBF7zA). In image captioning, a CNN encoder first extracts a fixed‑size feature vector from the image, which is then fed into the RNN decoder as its initial input. The decoder generates one word per time step, updating its hidden state after each token, thereby capturing temporal dependencies that shape fluent, context‑aware captions [Source](https://www.youtube.com/watch?v=kG2lAPBF7zA). This sequence‑to‑sequence paradigm remains essential because it explicitly models the order of words, a capability that pure CNNs lack when producing language outputs [Source](https://projectcentersinchennai.co.in/ieee-domains/ieee-image-captioning-projects). Evaluation typically relies on BLEU, METEOR, CIDEr, and SPICE scores, which are computed by libraries that support tokenization and vocabulary management [Source](https://projectcentersinchennai.co.in/ieee-domains/ieee-image-captioning-projects).

> **[IMAGE GENERATION FAILED]** CNN encoder feeding image features into an RNN decoder that generates words sequentially.
>
> **Alt:** Image captioning architecture diagram
>
> **Prompt:** A clear, technical diagram of an image captioning system: a CNN encoder extracting a fixed-size feature vector from an image, feeding that vector into an RNN decoder. The decoder produces words one at a time, updating its hidden state after each token. Show arrows from image to encoder, encoder to decoder, and decoder outputting words. Label the components: CNN Encoder, Feature Vector, RNN Decoder, Hidden State, Word Output. Use a clean, minimal style suitable for a tutorial.
>
> **Error:** module 'aiohttp' has no attribute 'ClientSession'


> **[IMAGE GENERATION FAILED]** Step‑by‑step flow of how an RNN generates words in image captioning.
>
> **Alt:** RNN word generation flowchart
>
> **Prompt:** A flowchart illustrating the RNN word generation process in image captioning. Start with an initial hidden state and a start token. Show the RNN computing a new hidden state, generating a word, feeding that word back as input, and repeating until an end token is produced. Use boxes for steps and arrows for data flow. Label each step: Initial Hidden State, Input Token, RNN Compute, Generate Word, Feed Back, End Token. Keep the design simple and technical.
>
> **Error:** module 'aiohttp' has no attribute 'ClientSession'
