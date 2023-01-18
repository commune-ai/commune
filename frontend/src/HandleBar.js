import React, { useState } from 'react';
import './HandleBar.css';

const HandleBar = () => {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  return (
    <div className="handle-bar">
        <div className="handle-bar-logo">
            <img src={''} alt="Logo" />
        </div>
        <div className="handle-bar-tabs">
            <div className="handle-bar-tab">Models</div>
            <div className="handle-bar-tab">Datasets</div>
            <div className="handle-bar-tab">Pipelines</div>
        </div>
        <div className="handle-bar-dropdown" onClick={() => setIsDropdownOpen(!isDropdownOpen)}>
            <div className="handle-bar-dropdown-label">Menu</div>
            {isDropdownOpen && (
                <div className="handle-bar-dropdown-content">
                    <div className="handle-bar-dropdown-item">Documentation</div>
                    <div className="handle-bar-dropdown-item">About</div>
                    <div className="handle-bar-dropdown-item">Contact</div>
                    <div className="handle-bar-dropdown-item">Social</div>
                </div>
            )}
        </div>
    </div>
  );
};

export default HandleBar;