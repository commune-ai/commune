import React, { useState } from 'react';
import Item from './Item';
import './Marketplace.css';

const Marketplace = (props) => {
    const [dropdownOpen, setDropdownOpen] = useState(false);

    return (
        <div className="marketplace">
            <div className="marketplace-header">
                <div className="marketplace-title">
                    <h1>Marketplace</h1>
                </div>
                <div className="marketplace-tabs">
                    <div className="tab large-round">Models</div>
                    <div className="tab large-round">Datasets</div>
                    <div className="tab large-round">Pipelines</div>

                    <div className="dropdown" onClick={() => setDropdownOpen(!dropdownOpen)}>
                        <div className="dropdown-title">Menu</div>
                        <div className={`dropdown-content ${dropdownOpen ? 'open' : ''}`}>
                            <a href="/documentation">Documentation</a>
                            <a href="/contact">Contact</a>
                            <a href="/about">About</a>
                    </div>
                </div>
                </div>
            </div>
            <div className="marketplace-items-container">
                <div className="marketplace-items">
                    {props.items.map((item, index) => {
                        return <Item key={index} item={item} />
                    })}
                </div>
            </div>
        </div>
    )
    }
    
export default Marketplace;
