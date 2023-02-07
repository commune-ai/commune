import React, { useState, useEffect } from 'react';
import Item from './Item';
import './Marketplace.css';
import axios from 'axios';


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


const Marketplace = (props) => {
    const [expanded, setExpanded] = useState(false);
    const [modules, setModules] = useState([]);


    useEffect(() => {
        const res = axios.post(`http://0.0.0.0:8000/list_modules` )
        .then(res => ( setModules(res.data) ))
    }, [])


    console.log(modules)

    const handleClick = () => {
        setExpanded(!expanded);
    }

    

    return (
        <div className="marketplace">
            <div className="marketplace-header">
                <div className="marketplace-title" onClick={handleClick}>
                    <h1>Marketplace</h1>
                </div>
                <div className={`sidebar ${expanded ? 'expanded' : ''}`}>
                    {/* sidebar content */}
                </div>
                <HandleBar></HandleBar>

            </div>
        
            <div className="marketplace-items-container">
                {modules.map(( module, index) => {
                    console.log(module)
                    return (
                        <Item key={module.name} {...module}></Item>         
                    )
                })}
            </div>
        </div>
    )
}

export default Marketplace;
