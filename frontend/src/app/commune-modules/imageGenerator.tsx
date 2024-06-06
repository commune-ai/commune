import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import debounce from 'lodash.debounce';
import { Prediction } from 'replicate';
import { ValidatorType } from '@/types';

interface Props {
    module?: ValidatorType;
    savedDescription: string;
    savedImageUrl: string
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const ImageGeneratorComponent: React.FC<Props> = ({ module, savedDescription, savedImageUrl }) => {

    const [imageUrl, setImageUrl] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [description, setDescription] = useState<string | null>(module?.description || null);
    const [prediction, setPrediction] = useState<Prediction | null>(null);
    const [isHovered, setIsHovered] = useState<boolean>(false);


    const fetchDescription = useCallback(async (name: string) => {
        try {
            const response = await axios.post('/api/generate-description', { name });
            if (response.status === 200) {
                setDescription(response.data.description);
                saveDescription(response.data.description, name)
            } else {
                setError('Failed to generate description');
            }
        } catch (error) {
            console.error('Error fetching description:', error);
            setError('Error fetching description');
        }
    }, []);

    const saveDescription = async (description: string, name: string) => {
        try {
            const response = await axios.post(`${process.env.NEXT_PUBLIC_BACKEND_URL}api/save-description/`, { description, name });
            if (response.status !== 200) {
                setError('Failed to save description');
            }
        } catch (error) {
            console.error('Error saving description:', error);
            setError('Error saving description');
        }
    };

    const generatePrediction = useRef(
        debounce(async (module: ValidatorType, description: string) => {
            if (module && !module.image && description) {
                try {
                    const response = await fetch('/api/stable-diffusion', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ prompt: description }),
                    });

                    const prediction = await response.json();
                    if (response.status !== 201) {
                        setError(prediction.error ?? 'An error occurred');
                        return;
                    }

                    setImageUrl(prediction.output[prediction.output.length - 1]);
                    console.log('------------This is the stable Image generation-----------', prediction.output[prediction.output.length - 1]);

                    saveDataToBackend(module.name, prediction.output[prediction.output.length - 1]);

                    // while (prediction.status !== 'succeeded' && prediction.status !== 'failed') {
                    //     await sleep(500);
                    //     const response = await fetch('/api/predictions/' + prediction.id);
                    //     prediction = await response.json();
                    //     if (response.status !== 200) {
                    //         setError(prediction.error ?? 'An error occurred');
                    //         return;
                    //     }
                    //     setPrediction(prediction);
                    // }

                    // if (prediction.status === 'succeeded' && prediction.output) {
                    //     setImageUrl(prediction.output[prediction.output.length - 1]);
                    //     // Save to backend
                    //     saveDataToBackend(module.name, imageUrl);
                    // }

                } catch (error) {
                    console.error('Error generating prediction:', error);
                }
            }
        }, 300)
    ).current;

    useEffect(() => {
        if (module && !module.description && module.name && !savedDescription) {
            fetchDescription(module.name);
        }
    }, [module, fetchDescription, savedDescription]);

    useEffect(() => {
        if (module && description && !savedImageUrl) {
            generatePrediction(module, description);
        }
    }, [module, description, generatePrediction, savedImageUrl]);

    //save imageurl and module name
    const saveDataToBackend = async (moduleName: string, imageUrl: string | null) => {
        try {
            const response = await axios.post(`${process.env.NEXT_PUBLIC_BACKEND_URL}api/save-data`, { moduleName, imageUrl });
            if (response.status !== 200) {
                setError('Failed to save data to backend');
            }
        } catch (error) {
            console.error('Error saving data to backend:', error);
            setError('Error saving data to backend');
        }
    };

    const finalImageUrl = module?.image
        ? `${process.env.NEXT_PUBLIC_ENDPOINT}/${module.image}`
        : imageUrl;

    return (
        <div className={`h-[340px] rounded-tl-[20px] rounded-tr-[20px] w-full relative space-y-2 transition-all duration-150 ease-out overflow-hidden ${finalImageUrl ? '' : 'bg-[#0a0e11]'} flex justify-center items-center rounded-3xl mx-auto`}
            style={{
                backgroundImage: finalImageUrl ? `url(${finalImageUrl})` : 'none',
                backgroundSize: 'cover',
                backgroundPosition: 'center',
            }}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
        >
            {
                !finalImageUrl &&
                <span className='flex items-center justify-center p-2 text-center'>
                    {(!finalImageUrl && !isHovered) ? module?.name : module?.description}
                </span>
            }

        </div>
    );
};

export default ImageGeneratorComponent;
