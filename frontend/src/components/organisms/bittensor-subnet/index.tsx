"use client"
import { useEffect, useRef, useState } from 'react';
import Image from 'next/image';
import meImage from '../../../../public/img/frontpage/chat-avatar.png';
import communeImage from '../../../../public/svg/commune.svg';
import Loading from '@/components/molecules/bittensor/loading';

type ConversationType = {
    role: string;
    content: string;
}

const BittensorSubnet = () => {
    const inputRef = useRef<HTMLInputElement | null>(null);
    const [conversation, setConversation] = useState<ConversationType[]>([]);
    const [inputVal, setInputValue] = useState<string>('');
    const [loading, setLoading] = useState<boolean>(false);

    useEffect(() => {
        function handleKeyPress(event: KeyboardEvent) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        const inputElement = inputRef.current!;

        if (inputElement) {
            inputElement.addEventListener('keypress', handleKeyPress);

            return () => {
                inputElement.removeEventListener('keypress', handleKeyPress);
            };
        }

    }, [inputVal]);


    const onSendQueryHandle = async () => {
        const data = {
            role: 'user',
            content: inputVal
        }

        setConversation(prev => [...prev, data]);

        setLoading(true);

        const response = await fetch('/api/bittensor', {
            method: 'POST',
            body: JSON.stringify(data)
        });

        if (response.status == 200) {
            const result = await response.json();
            const replyContent = result.data[0]?.choices[0]?.delta?.content;
            setConversation(prev => [...prev, { role: 'commune', content: replyContent }]);
        }
        setLoading(false);
    }


    const sendMessage = () => {
        if (inputVal == '') {
            return;
        }

        onSendQueryHandle();
        setInputValue('');
    }


    return (
        <div className='border-[2px] border-[#f2f2f2] rounded-[20px] sm:px-[10px] py-[60px] relative'>
            {
                loading && <Loading />
            }
            <div className='h-[640px] mb-[30px] subnet-chat-content overflow-y-scroll'>
                <div className='mx-[10px] sm:mx-[30px]'>
                    {
                        conversation.map((item, idx) => (
                            item.role == "user" ?
                                <div key={idx} className="flex justify-end items-start mb-4">
                                    <div className="mr-2 py-3 px-4 bg-blue-400 rounded-bl-3xl rounded-tl-3xl rounded-tr-xl text-white" >
                                        {item.content}
                                    </div>
                                    <Image src={meImage} width={50} height={50} alt='me' />
                                </div>
                                :
                                <div key={idx} className="flex justify-start items-start mb-4">
                                    <Image src={communeImage} width={50} height={50} alt='commune' />
                                    <div className="ml-2 py-3 px-4 bg-gray-400 rounded-br-3xl rounded-tr-3xl rounded-tl-xl text-white" >
                                        {item.content}
                                    </div>
                                </div>
                        ))
                    }
                </div>
            </div>
            <div className="absolute bottom-[20px] left-1/2 -translate-x-1/2 flex justify-center w-full px-[20px]">
                <div className='w-full sm:w-[540px] md:w-[640px] mx-auto relative'>
                    <input ref={inputRef} value={inputVal} onChange={({ target: { value } }) => setInputValue(value)} disabled={loading}
                        className="w-full bg-gray-500 py-5 px-3 rounded-xl outline-none 
                            hover:bg-[#303846] focus:bg-[#303846] duration-300 transition-all"
                        type="text" placeholder="Type your message here..."
                    />
                </div>
            </div>
        </div>
    )
}


export default BittensorSubnet;

