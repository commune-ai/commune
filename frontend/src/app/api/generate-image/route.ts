import { NextResponse } from 'next/server';
import axios from 'axios';

export async function POST(request: Request) {
    const { description } = await request.json();

    if (!description) {
        return NextResponse.json({ error: 'Description is required' }, { status: 400 });
    }

    try {
        const apiKey = process.env.NEXT_PUBLIC_DEEPAI_API_KEY;  
        // Ensure you have this in your .env file
        const response = await axios.post(
            'https://api.deepai.org/api/text2img',
            {
                headers: {
                    'api-key': apiKey,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: description,
                })
            }

        );

        const imageUrl = response.data.output_url;

        return NextResponse.json({ imageUrl });
    } catch (error) {

        return NextResponse.json({ error: 'Failed to generate image' }, { status: 500 });
    }
}
