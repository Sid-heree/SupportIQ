# Phase 0: Problem Understanding

## 1. What is a Ticket Queue?
A ticket queue is a logical grouping or bucket where incoming customer support issues are categorized based on the team, department, or specialized knowledge required to resolve them. 
* **Examples:** `Billing & Invoicing`, `Technical Support`, `Account Security`, `Product Feedback`.
* **Purpose:** It ensures that a ticket doesn't just float in a massive inbox, but immediately lands with the specific department trained to handle it.

## 2. What is Ticket Priority?
Ticket priority defines the urgency and business impact of a customer issue. It dictates the Service Level Agreement (SLA)—the strict time window within which the support team *must* respond and resolve the issue.
* **P1 (Critical / Urgent):** System-wide outages, data breaches, or core functionality failure affecting all users. (Requires immediate 24/7 attention).
* **P2 (High):** Significant performance degradation or a core feature broken for a large group of users with no immediate workaround.
* **P3 (Medium):** Standard technical issues, bugs with minor business impact, or features broken with an available workaround.
* **P4 (Low):** Minor cosmetic bugs, general inquiries, or feature requests.

## 3. What is Ticket Type?
Ticket type classifies the structural nature or intent of the incoming communication. Knowing the "type" determines what kind of playbook or workflow the automation or agent needs to follow.
* **Incident:** Something that used to work is now broken (e.g., "I can't log in").
* **Service Request:** The customer is asking for access, provision, or assistance with a standard action (e.g., "Please upgrade my account tier").
* **Inquiry / Question:** General questions about how to use the software (e.g., "How do I export my data to CSV?").
* **Problem:** Underneath multiple incidents, there is a root cause that engineering needs to fix (e.g., a buggy backend deployment causing thousands of login failures).

## 4. What is a Support Workflow?
A support workflow is the lifecycle of a ticket from creation to closure. A modern, automated lifecycle follows these stages:
1. **Intake / Ingestion:** A customer submits an issue via email, chat, or web form.
2. **Triage & Classification:** The incoming text is analyzed to determine its Queue, Priority, and Type. (Historically manual; automated using ML in SupportIQ).
3. **Routing:** The ticket is assigned to the correct queue/agent.
4. **Investigation & Resolution:** The agent investigates, pulls context (historical similar tickets), and drafts a response (leveraging RAG/LLMs).
5. **Closure & Feedback:** The ticket is resolved, and the customer provides a CSAT (Customer Satisfaction) score.

## 5. Why do companies use ticket routing?
Manual triage is one of the most expensive and slow components of customer service. Companies automate ticket routing to achieve:
* **Reduced Average Handle Time (AHT) & First Response Time (FRT):** Tickets go directly to the right person without manual hand-offs.
* **Lower Operational Costs:** Eliminates the need for a dedicated team of human "dispatchers" whose only job is reading and moving tickets.
* **Improved Customer Satisfaction (CSAT):** Faster, highly accurate resolutions lead to happier customers.
* **Agent Burnout Mitigation:** Agents only see tickets they are skilled to solve, reducing frustration and context-switching.