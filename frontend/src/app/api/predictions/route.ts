import { NextResponse } from 'next/server';
import Replicate from 'replicate';

// Assuming WebhookEventType is imported from Replicate or defined somewhere in your project.
type WebhookEventType = 'start' | 'completed';

const replicate = new Replicate({
    auth: process.env.NEXT_PUBLIC_REPLICATE_API_TOKEN as string,
});

// In production and preview deployments (on Vercel), the VERCEL_URL environment variable is set.
// In development (on your local machine), the NGROK_HOST environment variable is set.
const WEBHOOK_HOST = process.env.NEXT_PUBLIC_DEPLOYED_URL
   
export async function POST(request: Request) {
    if (!process.env.NEXT_PUBLIC_REPLICATE_API_TOKEN) {
        throw new Error(
            'The REPLICATE_API_TOKEN environment variable is not set. See README.md for instructions on how to set it.'
        );
    }

    const { prompt }: { prompt: string } = await request.json();

    const options: {
        version: string,
        input: { prompt: string },
        webhook?: string,
        webhook_events_filter?: WebhookEventType[]
    } = {
        version: '8beff3369e81422112d93b89ca01426147de542cd4684c244b673b105188fe5f',
        input: { prompt }
    };

    if (WEBHOOK_HOST) {
        options.webhook = `${WEBHOOK_HOST}/api/webhooks`;
        options.webhook_events_filter = ["start", "completed"];
    }

    const prediction = await replicate.predictions.create(options);

    if (prediction?.error) {
        return NextResponse.json({ detail: prediction.error }, { status: 500 });
    }

    return NextResponse.json(prediction, { status: 201 });
}
