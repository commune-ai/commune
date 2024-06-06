import React, { useState } from 'react';
import classes from './checkbox.module.css';

type CheckboxProps = {
    name: string;
    text: string;
    defaultValue: string | number | boolean;
};

export default function Checkbox({
    name,
    text,
    defaultValue,
}: CheckboxProps) {

    const [checked, setChecked] = useState(!!defaultValue);

    return (
        <label className={classes.container}>
            {text}
            <input
                className={classes.input}
                type="checkbox"
                name={name}
                checked={checked}
                onChange={() => setChecked(state => !state)}
            />
            <div className={classes.checkmark} />
        </label>
    );
}
