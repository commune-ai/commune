import React, { useState } from 'react';
// import Item from './Item';
import './Marketplace.css';


const HandleBar = (props) => {
    const [dropdownOpen, setDropdownOpen] = useState(false);

    return (
        <div className="handle-bar">
            <div className="handle-bar-tabs">
                <div className="tab">Models</div>
                <div className="tab">Datasets</div>
                <div className="tab">Pipelines</div>
            </div>
            <div className="dropdown" onClick={() => setDropdownOpen(!dropdownOpen)}>
                <div className="dropdown-title">Menu</div>
                <div className={`dropdown-content ${dropdownOpen ? 'open' : ''}`}>
                    <a href="/documentation">Documentation</a>
                    <a href="/contact">Contact</a>
                    <a href="/about">About</a>
                </div>
            </div>
        </div>
    )
}


const Item = (props) => {
    
    const [expanded, setExpanded] = useState(false);

    const handleClick = () => {
        setExpanded(!expanded);
    }

    return (
        <div className={`marketplace-item tab ${expanded ? 'expanded' : ''}`} onClick={handleClick}>
            {/* item content */}
        </div>
    )
}

const Marketplace = (props) => {


    return (
        <div className="marketplace">
            <div className="marketplace-header">
                <div className="marketplace-title">
                    <h1>Marketplace</h1>
                </div>
                {/* <div className="marketplace-tabs">
                    <div className={`tab ${props.selectedTab === 'models' ? 'selected' : ''}`} onClick={() => props.setSelectedTab('models')}>Models</div>
                    <div className={`tab ${props.selectedTab === 'datasets' ? 'selected' : ''}`} onClick={() => props.setSelectedTab('datasets')}>Datasets</div>
                    <div className={`tab ${props.selectedTab === 'pipelines' ? 'selected' : ''}`} onClick={() => props.setSelectedTab('pipelines')}>Pipelines</div>
                </div> */}
                <HandleBar></HandleBar>

            </div>

            <div className="marketplace-items-container">
                {props.items.map((item, index) => {
                    return (
                        <Item></Item>         
                    )
                })}
            </div>
        </div>
    )
}

export default Marketplace;
