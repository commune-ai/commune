import React, { useState } from 'react';
import Item from './Item';
import './Marketplace.css';
import HandleBar from './HandleBar';



const Marketplace = (props) => {
    const [expandedItem, setExpandedItem] = useState(null);


    return (
        <div className="marketplace">
            <div className="marketplace-header">
                <div className="marketplace-title">
                    <h1>Marketplace</h1>
                </div>
                <div className="marketplace-tabs">
                    <div className={`tab ${props.selectedTab === 'models' ? 'selected' : ''}`} onClick={() => props.setSelectedTab('models')}>Models</div>
                    <div className={`tab ${props.selectedTab === 'datasets' ? 'selected' : ''}`} onClick={() => props.setSelectedTab('datasets')}>Datasets</div>
                    <div className={`tab ${props.selectedTab === 'pipelines' ? 'selected' : ''}`} onClick={() => props.setSelectedTab('pipelines')}>Pipelines</div>
                </div>
            </div>

            <div className="marketplace-items-container">
                {props.items.map((item, index) => {
                    return (
                        <div 
                            key={index}
                            className={`marketplace-item tab ${expandedItem === item ? 'expanded' : ''}`}
                            onClick={() => {
                                if(expandedItem === item) {
                                    setExpandedItem(null);
                                } else {
                                    setExpandedItem(item);
                                }
                            }
                            }
                        >
                            {item}
                        </div>
                    )
                })}
            </div>
        </div>
    )
}

export default Marketplace;
