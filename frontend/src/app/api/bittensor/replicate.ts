import { NextApiRequest, NextApiResponse } from 'next';
import axios from "axios";

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const cursor: string | string[] | undefined = req && req?.query?.cursor;

  if (cursor === '') {
    try {
      const response = await axios.get('https://api.replicate.com/v1/models', {
        headers: {
          'Content-Type': 'application/json',
          Authorization: 'Token r8_ZGZlzThfRkPZVDMygVclY1XZ9AuxmIQ2qwwPP',
          "Access-Control-Allow-Headers": "Content-Type",
          "Access-Control-Allow-Origin": '**',
          "Access-Control-Allow-Methods": "OPTIONS,POST,GET,PATCH"
        },
      })
      res.status(200).json({ modules: response.data })
    } catch (err) {
      res.status(500).json({ error: err });
    }
  }
  else {
    try {
      const response = await axios.get(`${cursor}`, {
        headers: {
          'Content-Type': 'application/json',
          Authorization: 'Token r8_ZGZlzThfRkPZVDMygVclY1XZ9AuxmIQ2qwwPP',
          "Access-Control-Allow-Headers": "Content-Type",
          "Access-Control-Allow-Origin": '**',
          "Access-Control-Allow-Methods": "OPTIONS,POST,GET,PATCH"
        },
      })
      return res.status(200).json({ modules: response.data })
    } catch (err) {
      return res.status(500).json({ error: err });
    }
  }
}
