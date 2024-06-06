"use client"
import React from 'react';
import classnames from 'classnames';
import Image from "next/image"
import Link from "next/link"
import { MdVerified } from "react-icons/md";
interface ModuleCardInterface {
    url: string
    name: string
    image: string
    description: string
    registerKey: string
    verified: boolean
    tags: string[]
}


const ModuleCard = ({ url, name, image, description, registerKey, verified, tags }: ModuleCardInterface) => {

    console.log('--------This is the status--------', verified);

    return (
        <div>
            {registerKey &&
                <div className="card card-compact w-full shadow-xl p-4">
                    <div className="mockup-browser border rounded-2xl bg-gray-900 h-full w-full">
                        <div className="inset-0 rounded-2xl bg-opacity-50 backdrop-filter backdrop-blur-lg flex items-center justify-center">
                            <div className="text-white text-2xl font-bold">
                                <Link href={url} className="input">
                                    {name}
                                </Link>
                            </div>
                        </div>
                        <div className="mockup-browser-toolbar">
                            <Link href={url} className="input">
                                <u>{url}</u>
                            </Link>
                        </div>
                        <Link href={url} >

                            <div className="justify-center card-title">{name}
                                <div className="tooltip tooltip-top" data-tip="verified">
                                    <MdVerified className={`${name === 'ComHub' ? 'text-yellow-500' : 'text-blue-500'}`} />
                                </div>
                            </div>
                            <div className="flex justify-center px-4 py-16 bg-gray-900">
                                <div className="relative min-h-40 min-w-40">
                                    <Image fill src={image} alt="module-image" />
                                </div>
                            </div>
                        </Link>

                        <div className="card-body py-0">
                            <div className="flex gap-2">


                                {tags.map((tag) => (
                                    <div key={tag} className={classnames('badge', 'badge-outline', {
                                        'badge-accent': tag === 'stats',
                                        'badge-primary': tag === 'staking',
                                        'badge-info': tag === 'bridge',
                                        'badge-secondary': tag === 'wallet',
                                        'badge-success': tag === 'chat' || tag === 'GPT' || tag === 'events',
                                        'badge-warning': tag === "hub",
                                        'badge-error': tag === "new",
                                        'badge-default': tag === 'com' || tag === 'coming soon' || tag === 'Ai'
                                    })}>{tag}</div>
                                ))}

                            </div>

                            <div role="alert" className="alert">
                                <div className="card-text font-500">
                                    {description}
                                </div>
                            </div>
                            <div className={classnames('divider justify-center flex mt-4 mb-0')}>Register Key</div>
                            <p className="text-xs text-slate-600 m-auto">{registerKey}</p>
                        </div>
                    </div>
                </div>
            }
        </div>
    )
}

export default ModuleCard
