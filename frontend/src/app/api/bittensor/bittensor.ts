import { NextApiRequest, NextApiResponse } from 'next';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {

    const data = JSON.parse(req.body);

    const key = process.env.NEXT_PUBLIC_BITTENSOR_KEY!;


    try {
        const response = await fetch('https://api.corcel.io/v1/text/cortext/chat', {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Authorization': key,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: 'cortext-ultra',
                stream: false,
                top_p: 1,
                temperature: 0.0001,
                messages: [
                    data
                ],
                miners_to_query: 1,
                top_k_miners_to_query: 40,
                ensure_responses: true
            })
        })


        const result = await response.json();

        res.status(200).json({ data: result })

    } catch (error) {
        console.log(error)
        return res.status(500).json({ error: error });
    }
}


