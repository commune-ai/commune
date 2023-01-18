import React from 'react';
import Marketplace from './Marketplace';

const items = Array.from({length: 100}, (_, i) => { name: `item ${i}`});


const App = () => {
    return (
        <div className="app">
            <Marketplace items={items} />
        </div>
    )
}

export default App;