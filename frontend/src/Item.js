import React, { useState } from 'react';
import './Item.css';

const Item = (props) => {
    const [expanded, setExpanded] = useState(false);
    const [status, setStatus] = useState("online");

    const handleClick = () => {
        setExpanded(!expanded);
    }

    return (
        <div className={`item ${expanded ? 'expanded' : ''}`} onClick={handleClick}>
            <div className="item-title-container">
                <h2 className="item-title">{props.item.title}</h2>
                <div className={`status-indicator ${status === 'online' ? 'online' : 'offline'}`}>
                    {status}
                </div>
            </div>
            <div className="item-description">
                {props.item.description}
            </div>
            <div className="item-attributes-container">
                <div className="item-attributes">
                {props.item.attributes.map((attribute, index) => {
                    return <div className="attribute-tag" key={index}>{attribute}</div>
                })}
                </div>
            </div>
        </div>
    )
}

export default Item;
