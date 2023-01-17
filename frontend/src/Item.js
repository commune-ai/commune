import React from 'react';
import './Item.css';

const Item = (props) => {
    return (
        <div className="item">
            {/* <img src={props.item.image} alt={props.item.name} /> */}
            <h3>{props.item.name}</h3>
            <p>{props.item.price}</p>
            {/* <button>Add to cart</button> */}
        </div>
    )
}

export default Item;