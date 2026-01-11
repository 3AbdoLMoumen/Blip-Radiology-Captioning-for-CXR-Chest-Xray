def predict(image, processor, model, max_length=64, top_p=0.9, temperature=0.5, do_sample=True):
    inputs = processor(images=image, return_tensors="pt").to("cpu")

    pixel_values = inputs.pixel_values
    generated_ids = model.generate(
        pixel_values=inputs.pixel_values,
        max_length=max_length,
        do_sample=do_sample,   
        top_p=top_p,
        temperature=temperature
    )
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption