import os
import requests
import base64
import commune as c

class StabilityAI(c.Module):
    def __init__(self, engine_id="stable-diffusion-xl-1024-v1-0", api_host="https://api.stability.ai", api_key=None):
        self.api_key = os.getenv("STABILITY_API_KEY", api_key)
        if api_key==None:
            self.api_key = c.choice(self.api_keys())
        self.api_host = os.getenv("API_HOST", api_host)
        self.engine_id = engine_id
        assert self.api_key is not None, "Missing Stability API key."

    def account(self):
        url = f"{self.api_host}/v1/user/account"
        response = self._make_request("GET", url)
        return response.json()

    def image2image(self, init_image_path, text_prompt, image_strength=0.35, cfg_scale=7, samples=1, steps=30):
        url = f"{self.api_host}/v1/generation/{self.engine_id}/image-to-image"
        files = {"init_image": open(init_image_path, "rb")}
        data = {
            "image_strength": image_strength,
            "init_image_mode": "IMAGE_STRENGTH",
            "text_prompts[0][text]": text_prompt,
            "cfg_scale": cfg_scale,
            "samples": samples,
            "steps": steps,
        }
        response = self._make_request("POST", url, files=files, data=data)
        return self._process_response(response)

    def text2image(self, text_prompt, cfg_scale=7, height=1024, width=1024, samples=1, steps=30):
        url = f"{self.api_host}/v1/generation/{self.engine_id}/text-to-image"
        
        is_batch = isinstance(text_prompt, list)

        
        json = {
            "text_prompts": [{"text": text_prompt}],
            "cfg_scale": cfg_scale,
            "height": height,
            "width": width,
            "samples": samples,
            "steps": steps,
        }
        response = self._make_request("POST", url, json=json)
        result = self._process_response(response)
        if is_batch:
            return result
        else:
            return result[0]

    def upscale_image(self, image_path, width=1024):
        url = f"{self.api_host}/v1/generation/esrgan-v1-x2plus/image-to-image/upscale"
        files = {"image": open(image_path, "rb")}
        data = {"width": width}
        response = self._make_request("POST", url, files=files, data=data)
        return response.content

    def image_masking(self, init_image_path, mask_image_path, text_prompt, cfg_scale=7, clip_guidance_preset="FAST_BLUE", samples=1, steps=30):
        url = f"{self.api_host}/v1/generation/{self.engine_id}/image-to-image/masking"
        files = {
            'init_image': open(init_image_path, 'rb'),
            'mask_image': open(mask_image_path, 'rb'),
        }
        data = {
            "mask_source": "MASK_IMAGE_BLACK",
            "text_prompts[0][text]": text_prompt,
            "cfg_scale": cfg_scale,
            "clip_guidance_preset": clip_guidance_preset,
            "samples": samples,
            "steps": steps,
        }
        response = self._make_request("POST", url, files=files, data=data)
        return self._process_response(response)

    def _make_request(self, method, url, **kwargs):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if 'headers' in kwargs:
            headers.update(kwargs['headers'])
        response = requests.request(method, url, headers=headers, **kwargs)
        if response.status_code != 200:
            raise Exception(f"Non-200 response: {response.text}")
        return response

    def _process_response(self, response):
        data = response.json()
        images = []
        for i, image in enumerate(data["artifacts"]):
            images.append(base64.b64decode(image["base64"]))
        return images


    def test(self):

        import tempfile
        # Initialize the StabilityAI class
        self.stability_ai = StabilityAI()
        # Create a temporary directory for test files
        self.test_dir = tempfile.TemporaryDirectory()
        # Call the account method
        try:
            result = self.account()
            assert isinstance(result, dict)  # Check if the result is a dictionary
            responses = [self.test_image2image(), self.test_text2image()]
            self.test_dir.cleanup()
        except Exception as e:
            self.test_dir.cleanup()
            raise e
        return responses


    def test_image2image(self):
        # Create a temporary image file for testing
        temp_image_path = os.path.join(self.test_dir.name, "temp_image.png")
        with open(temp_image_path, "wb") as f:
            # Create a small black image for testing
            f.write(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcAAgUB/gc1KgkAAAAASUVORK5CYII="))
        
        # Call the image2image method
        result = self.stability_ai.image2image(temp_image_path, "Test prompt")
        assert isinstance(result, list)  # Check if the result is a list
        assert isinstance(all(isinstance(item, bytes) for item in result))  # Check if all items in the list are bytes

    def test_text2image(self):
        # Call the text2image method
        result = self.stability_ai.text2image("Test prompt")
        assert isinstance(result, list)
        assert all(isinstance(item, bytes) for item in result)

    def text2video(self, path):
        assert os.path.exists(path), f'Path {path} does not exist'
        import requests
        path = self.resolve_path(path)

        response = requests.post(
            "https://api.stability.ai/v2alpha/generation/image-to-video",
            headers={
                f"authorization": "Bearer {self.api_key}",
            },
            data={
                "seed": 0,
                "cfg_scale": 2.5,
                "motion_bucket_id": 40
            },
            files={
                "image": ("file", open(path, "rb"), )
            },
        )

        if response.status_code != 200:
            raise Exception("Non-200 response: " + str(response.text))

        data = response.json()
        generation_id = data["id"]
        return data

    # Additional tests for other methods can be added here
