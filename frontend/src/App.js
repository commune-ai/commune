import React from 'react';
import Marketplace from './Marketplace';

const items = Array.from({length: 100}, (_, i) => ({item: {title: `Module ${i}`,attributes: ['metrics','bro'], description: 'Description'}}));

const App = () => {
    return (
        <div className="app">
            <Marketplace items={items} />
        </div>
    )
}

export default App;