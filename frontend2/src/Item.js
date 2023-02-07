import React, { useState, useEffect } from 'react';
import './Item.css';
import axios from 'axios';

const Item = (props) => {
    const [expanded, setExpanded] = useState(false);
    const [status, setStatus] = useState("online");
    const [description, setDescription] = useState("");

    console.log(props);


    const handleClick = () => {
        setExpanded(!expanded);
    }

    return (
        <div className={`item ${expanded ? 'expanded' : ''}`} onClick={handleClick}>
            <div className="item-title-container">
                <h2 className="item-title">{props.name}</h2>
                <div className={`status-indicator ${status === 'online' ? 'online' : 'offline'}`}>
                    {status}
                </div>
            </div>
            <div className="item-description">
                {description}
            </div>
            {/* <div className="item-attributes-container">
                <div className="item-attributes">
                {props.tasks.map((attribute, index) => {
                    return <div className="attribute-tag" key={index}>{attribute}</div>
                })}
                </div>
            </div> */}
        </div>
    )
            }


export default Item;
