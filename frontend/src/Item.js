import React, { useState } from 'react';
import './Item.css';


const Item = (props) => {
    const [expanded, setExpanded] = useState(false);

    const handleClick = () => {
        setExpanded(!expanded);
    }

    return (
        <div className={`item ${expanded ? 'expanded' : ''}`} onClick={handleClick}>
            {props.name}
        </div>
    )
}


export default Item;