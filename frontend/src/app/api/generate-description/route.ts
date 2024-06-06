import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';

const openai = new OpenAI({
    apiKey: process.env.NEXT_PUBLIC_API_KEY,
});

// IMPORTANT! Set the runtime to edge
export const runtime = 'edge';

export async function POST(request: NextRequest) {
    try {
        const { name } = await request.json();

        if (!name) {
            return NextResponse.json({ error: 'Name is required' }, { status: 400 });
        }

        // Add a system message to instruct the model to focus on providing SQL queries
        const newMessages = [
            {
                role: 'system',
                content: `Generate a description for a module named "${name}" about communeai.
                Commune is a protocol that aims to connect all developer tools into one network, fostering a more shareable, reusable, and open economy. It follows an inclusive design philosophy that is based on being maximally unopinionated. This means that developers can leverage Commune as a versatile set of tools alongside their existing projects and have the freedom to incorporate additional tools they find valuable.

                By embracing an unopinionated approach, Commune acknowledges the diverse needs and preferences of developers. It provides a flexible framework that allows developers to integrate specific tools seamlessly while avoiding imposing rigid structures or constraints. This adaptability enables developers to leverage Commune's capabilities in a manner that best aligns with their individual projects and workflows.
                
                The overarching goal of Commune is to create a collaborative ecosystem where developers can easily share, connect, and extend their tools, ultimately fostering innovation and efficiency within the development community. By providing a network that encourages openness and accessibility, Commune empowers developers to leverage the collective knowledge and resources of the community to enhance their own projects.
                You have to respond within 20 words.`
            },
            {
                role: 'user',
                content: `Module name: "${name}"`
            }
        ];

        // Ask OpenAI for a streaming chat completion given the prompt
        const chatCompletion = await openai.chat.completions.create({
            model: 'gpt-3.5-turbo',
            messages: newMessages as OpenAI.ChatCompletionMessageParam[],
        });

        const description = chatCompletion.choices[0].message.content?.trim();

        return NextResponse.json({ description });
    } catch (error) {
        console.error('Error generating description:', error);
        return NextResponse.json({ error: `Error generating description${error}` }, { status: 500 });
    }
}
