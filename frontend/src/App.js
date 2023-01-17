import React from 'react';
import Marketplace from './Marketplace';

const items = [
    { name: 'item 1', image: 'item1.jpg', price: '$10' },
    { name: 'item 2', image: 'item2.jpg', price: '$20' },
    { name: 'item 3', image: 'item3.jpg', price: '$30' },
    { name: 'item 3', image: 'item3.jpg', price: '$30' },
    { name: 'item 3', image: 'item3.jpg', price: '$30' },
    { name: 'item 3', image: 'item3.jpg', price: '$30' },
    { name: 'item 3', price: '$30' },
    { name: 'item 3', image: 'item3.jpg', price: '$30' }

];

const App = () => {
    return (
        <div className="app">
            <Marketplace items={items} />
        </div>
    )
}

export default App;