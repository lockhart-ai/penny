# Penny Feature TODO

This document tracks potential new features and enhancements for Penny.

## New Agent Types

### ReminderAgent
- Parse user requests like "remind me about X in 2 hours"
- Store reminders in database with scheduled execution time
- Send proactive reminders via channel
- Could use natural language parsing for time expressions

### TrendingAgent
- Monitor topics from recent conversations
- Search for breaking news related to discussed topics
- Send updates when significant developments occur
- Configurable frequency and relevance threshold

### DigestAgent
- Daily/weekly summary of conversations
- Highlights interesting discoveries or shared links
- Aggregates topics discussed across all users
- Scheduled delivery time (e.g., "morning digest")

### FactCheckAgent
- Background verification of claims made in conversations
- Uses multiple search sources for cross-referencing
- Gentle corrections if information appears outdated/incorrect
- Stores verified facts for future reference

## Enhanced Search & Information

### Multi-Source Search
- Add specialized search tools beyond Perplexity
- Wikipedia for encyclopedic content
- arXiv for academic papers
- GitHub for code examples
- YouTube for video content
- News-specific APIs for current events

### Media Handling
- Download and analyze images/PDFs shared by users
- OCR for text extraction from images
- Summarize linked articles/videos
- Generate image descriptions for accessibility

### Location Awareness
- Optional user location tracking
- Local search results (restaurants, events, weather)
- Time-zone aware scheduling
- Regional news and content

## User Interaction Features

### Multi-Turn Clarification
- Ask follow-up questions when requests are ambiguous
- Store clarification preferences per user
- Learn from past interactions to reduce questions

### Reaction-Based Feedback
- Users react to messages (👍/👎) for quality feedback
- Adjust response style based on reaction patterns
- Store feedback for future model fine-tuning
- Use reactions to prioritize topics for FollowupAgent

### Command System
- Special commands for power users: `/search`, `/summarize`, `/profile`, `/remind`
- Toggle features on/off per conversation
- Configure personal preferences inline

## Content Management

### Bookmark System
- Users can "save" interesting responses or links
- Retrieve saved content with search or tags
- Export bookmarks to other formats
- Share collections with other users

### Knowledge Base
- Build persistent knowledge from conversations
- Extract facts and store in structured format
- Reference past conversations: "What did we discuss about X?"
- Entity tracking (people, places, concepts mentioned)

### Quote Library
- Save interesting quotes or snippets from conversations
- Random quote sharing as discovery content
- Tag and categorize quotes by topic

## Platform Integrations

### Slack Support
- New SlackChannel implementation
- Thread-based conversations
- Workspace-aware context

### Telegram Support
- TelegramChannel via python-telegram-bot
- Group chat support with @mentions

### Matrix Support
- Open protocol alternative to proprietary platforms
- Self-hosted federation support

### Email Channel
- Long-form email responses
- Digest mode for batch processing
- Rich HTML formatting

## Intelligence & Memory

### Conversation Analytics
- Track topics of interest per user
- Identify conversation patterns
- Suggest related topics to explore
- Visualize discussion trends over time

### Cross-User Learning
- Share knowledge between users (with privacy controls)
- "User A also asked about this topic"
- Collaborative discovery (what's trending across all users)

### Semantic Search
- Vector embeddings for message history
- Find similar past conversations
- Recommend related threads

### Memory Layers
- Short-term: recent conversation context
- Medium-term: weekly/monthly themes
- Long-term: enduring interests and preferences
- Episodic: specific events or milestones

## Coordination & Scheduling

### Group Features
- Multi-user conversation support
- Coordinate responses when multiple people are active
- Mention/tag specific users
- Group profiles and shared interests

### Quiet Hours
- User-configurable do-not-disturb schedules
- Suppress background agents during quiet hours
- Queue messages for delivery at appropriate times

### Priority System
- Urgent vs. casual message classification
- Fast-track time-sensitive requests
- Deprioritize background tasks when user is active

## Developer & Debug Features

### Explain Mode
- Show reasoning steps and tool calls to users
- Transparency mode for debugging responses
- Display which sources were used and why

### A/B Testing Framework
- Test different prompts or models
- Compare response quality
- User preference collection
- Gradual rollout of new features

### Performance Metrics
- Response time tracking
- Token usage monitoring
- Search API quota management
- Alert on degraded performance

## Fun & Engagement

### Personality Modes
- Switch between different response styles
- Technical/casual/humorous/professional modes
- Per-user or per-conversation preferences

### Game/Quiz Features
- Spontaneous trivia based on conversation topics
- Daily challenges or brain teasers
- Track scores and streaks

### Easter Eggs
- Hidden features triggered by specific phrases
- Seasonal variations in responses
- Special responses for milestones (100th message, etc.)

---

## Most Impactful Quick Wins

These features would provide the highest value with reasonable implementation effort:

1. **ReminderAgent** - high utility, straightforward implementation
2. **Multi-Source Search** - immediate quality improvement
3. **Reaction-Based Feedback** - better learning without explicit user effort
4. **Bookmark System** - useful content organization
5. **Quiet Hours** - better respect for user preferences
