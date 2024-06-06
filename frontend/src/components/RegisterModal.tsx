import React from "react";
import { Modal, Form, Input, Button, Select } from "antd";
import { toast } from "react-toastify";
import modules from "../../modules.json"
import './registerModal.css'
interface ModalProps {
  onClose: () => void;
}

const RegisterModal: React.FC<ModalProps> = ({ onClose }) => {
  const [form] = Form.useForm();

  const handleSubmit = () => {
    toast.success('success')
    form.validateFields().then((values) => {
      // console.log(values.email);
      modules.push({
        "name": values.name,
        "url": values.url,
        "image": "/img/frontpage/comai-logo.png",
        "description": "Telemetry of Commune Ai",
        "registerKey": "5EJ9AUpSGafWeagdP5nwc5AwcYBkagYSZyx2BmLKWJrGBZUZ",
        "verified": false,
        "tags": [
          "stats",
          "staking",
          "wallet"
        ]
      })
      onClose(); // Close the modal
    });
  };

  return (
    <Modal
      title="Register Module"
      open={true}
      onCancel={onClose}
      styles={{
        mask: {
          backdropFilter: 'blur(10px)'
        }
      }
      }
      footer={[
        <Button key="cancel" onClick={onClose} className="bg-green-400">
          Cancel
        </Button>,
        <Button key="submit" type="primary" onClick={handleSubmit} className="bg-blue-800">
          Submit
        </Button>,
      ]}
    >
      <Form form={form}
        labelCol={{ span: 6 }}
        layout="vertical"
        // disabled={componentDisabled}
        style={{ maxWidth: 600 }}>
        <Form.Item
          name="title"
          label="Title"
          rules={[{ required: true, message: "Please enter Title" }]}
        >
          <Input />
        </Form.Item>
        <Form.Item
          name="emoji"
          label="Emoji"
          rules={[
            { required: true, message: "Please enter Emoji" },
          ]}
        ><Input />
        </Form.Item>
        <Form.Item
          name="colorfrom"
          label="ColorFrom"
          rules={[
            { required: true, message: "Please enter ColorFrom" },
          ]}
        >
          <Input />
        </Form.Item>
        <Form.Item
          name="colorto"
          label="ColorTo"
          rules={[
            { required: true, message: "Please enter ColorTo" },
          ]}
        >
          <Input />
        </Form.Item>
        <Form.Item
          name="sdk"
          label="SDK"
          rules={[
            { required: true, message: "Please enter SDK" },
          ]}
        >
          <Select
            options={[
              { value: 'Streamlit', label: 'Streamlit' },
              { value: 'Gradio', label: 'Gradio' },
              { value: 'Docker', label: 'Docker' },
              { value: 'Static', label: 'Static' },
            ]} />
        </Form.Item>
        <Form.Item
          name="app_file"
          label="App_File"
          rules={[
            { required: true, message: "Please enter App_File" },
          ]}
        >
          <Input />
        </Form.Item>
        <Form.Item
          name="pinned"
          label="Pinned"
          rules={[
            { required: true, message: "Please enter Pinned" },
          ]}
        >
          <Input />
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default RegisterModal;